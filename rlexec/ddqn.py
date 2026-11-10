"""Double DQN for the execution environment.

- State: market features (standardised with training-split statistics only) + time remaining
  + inventory remaining.
- Training reward: the variance-reduced "vs_twap" reward (see env.py). Evaluation always uses
  the plain shortfall.
- Model selection: the checkpoint with the best validation result is kept. The test split is
  never touched here.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .baselines import Policy
from .env import ExecutionEnv, run_episode


@dataclass
class DDQNConfig:
    hidden: list = field(default_factory=lambda: [64, 64])
    dueling: bool = True
    lr: float = 3e-4
    gamma: float = 1.0
    batch_size: int = 256
    buffer_size: int = 200_000
    epochs: int = 12
    parallel_envs: int = 16
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.6
    tau: float = 0.005
    warmup_steps: int = 5_000
    reward_scale: float = 0.1


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden, dueling: bool):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.dueling = dueling
        self.adv = nn.Linear(d, n_actions)
        self.val = nn.Linear(d, 1) if dueling else None

    def forward(self, x):
        h = self.body(x)
        a = self.adv(h)
        return self.val(h) + a - a.mean(dim=1, keepdim=True) if self.dueling else a


class Normalizer:
    """Standardises the market features with training-day statistics; private features pass through."""

    def __init__(self, days: list, n_private: int = 2):
        x = np.concatenate([d.features for d in days])
        self.mean = np.concatenate([x.mean(0), np.zeros(n_private)])
        self.std = np.concatenate([x.std(0) + 1e-8, np.ones(n_private)])

    def __call__(self, obs):
        return np.clip((obs - self.mean) / self.std, -10.0, 10.0).astype(np.float32)


class DDQNPolicy(Policy):
    def __init__(self, net: QNet, norm: Normalizer, device="cpu", name: str = "ddqn"):
        self.net, self.norm, self.device, self.name = net, norm, device, name

    @torch.no_grad()
    def act(self, obs, env):
        q = self.net(torch.as_tensor(self.norm(obs), device=self.device).unsqueeze(0))
        return env.action_frac(int(q.argmax(dim=1).item()))


def _mean_shortfall(env: ExecutionEnv, days: list, policy) -> np.ndarray:
    return np.array([run_episode(env, d, policy).is_bps for d in days])


def train_ddqn(env: ExecutionEnv, train_days: list, val_days: list, cfg: DDQNConfig, seed: int = 0,
               device: str | None = None, log=print):
    """Returns (policy, history). `env` is the evaluation env; training envs are built from it."""
    from .baselines import TWAP

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    make = lambda: ExecutionEnv(env.market, env.costs, env.parent, env.action, reward="vs_twap")
    envs = [make() for _ in range(cfg.parallel_envs)]
    B, n, nA = cfg.parallel_envs, train_days[0].n, env.n_actions
    obs_dim = len(env.obs_names)
    norm = Normalizer(train_days)

    net = QNet(obs_dim, nA, cfg.hidden, cfg.dueling).to(device)
    target = copy.deepcopy(net).requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    N = cfg.buffer_size
    buf_s, buf_s2 = np.zeros((N, obs_dim), np.float32), np.zeros((N, obs_dim), np.float32)
    buf_a, buf_r, buf_d = np.zeros(N, np.int64), np.zeros(N, np.float32), np.zeros(N, np.float32)
    ptr = size = 0

    batches_per_epoch = int(np.ceil(len(train_days) / B))
    total_steps = cfg.epochs * batches_per_epoch * n
    decay_steps = max(1, int(cfg.eps_decay_frac * total_steps))
    step = 0

    val_twap = _mean_shortfall(env, val_days, TWAP())
    policy = DDQNPolicy(net, norm, device)
    best = {"score": np.inf, "state": copy.deepcopy(net.state_dict()), "epoch": 0}
    history, t0 = [], time.time()

    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(len(train_days))
        losses = []
        for b in range(batches_per_epoch):
            idx = order[b * B:(b + 1) * B]
            if len(idx) < B:
                idx = np.concatenate([idx, rng.integers(0, len(train_days), B - len(idx))])
            s = np.stack([norm(e.reset(train_days[i])) for e, i in zip(envs, idx)])
            for t in range(n):
                eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * max(0.0, 1.0 - step / decay_steps)
                with torch.no_grad():
                    greedy = net(torch.as_tensor(s, device=device)).argmax(dim=1).cpu().numpy()
                a = np.where(rng.random(B) < eps, rng.integers(0, nA, B), greedy)
                s2 = np.zeros_like(s)
                for k, e in enumerate(envs):
                    o, r, done, _ = e.step(e.action_frac(int(a[k])))
                    if not done:
                        s2[k] = norm(o)
                    j = (ptr + k) % N
                    buf_s[j], buf_a[j], buf_r[j], buf_s2[j], buf_d[j] = s[k], a[k], r * cfg.reward_scale, s2[k], float(done)
                ptr, size, step = (ptr + B) % N, min(N, size + B), step + 1
                s = s2
                if size >= cfg.warmup_steps:
                    j = rng.integers(0, size, cfg.batch_size)
                    S, A = torch.as_tensor(buf_s[j], device=device), torch.as_tensor(buf_a[j], device=device)
                    R, S2 = torch.as_tensor(buf_r[j], device=device), torch.as_tensor(buf_s2[j], device=device)
                    D = torch.as_tensor(buf_d[j], device=device)
                    with torch.no_grad():  # double DQN: online net picks the action, target net values it
                        a2 = net(S2).argmax(dim=1, keepdim=True)
                        y = R + cfg.gamma * (1.0 - D) * target(S2).gather(1, a2).squeeze(1)
                    loss = nn.functional.smooth_l1_loss(net(S).gather(1, A.unsqueeze(1)).squeeze(1), y)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                    opt.step()
                    with torch.no_grad():
                        for p, pt in zip(net.parameters(), target.parameters()):
                            pt.mul_(1.0 - cfg.tau).add_(cfg.tau * p)
                    losses.append(loss.item())
        score = float((_mean_shortfall(env, val_days, policy) - val_twap).mean())
        history.append({"epoch": epoch, "val_vs_twap": round(score, 4),
                        "loss": round(float(np.mean(losses)), 6) if losses else None})
        if score < best["score"]:
            best = {"score": score, "state": copy.deepcopy(net.state_dict()), "epoch": epoch}
        log(f"  seed {seed} epoch {epoch:>2}/{cfg.epochs}  val vs TWAP {score:+8.2f} bps  "
            f"(best {best['score']:+.2f} @ {best['epoch']})  eps {eps:.2f}  {time.time() - t0:5.0f}s")

    net.load_state_dict(best["state"])
    net.eval()
    return policy, {"seed": seed, "best_epoch": best["epoch"], "best_val_vs_twap": round(best["score"], 4),
                    "epochs": history}
