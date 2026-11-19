"""The network: a transformer over the entities of the game.

Tokens: the 100 routes, the 36 cities, the 30 destination tickets, one per seat (5, relative to the
observer), one global token, the 5 face-up slots and the last 32 public events - 209 in all. The
board is in the attention: a learned bias per head for every pair of tokens by their relation on the
map (a route and its cities, routes that share a city, the two routes of a double, the distance in
hops between two cities, a ticket and its ends, a ticket and the routes at its ends), so every layer
sees the whole board with its structure.

The outputs (`DeepTTRNet.forward`), each read from the tokens it is about, never from one pooled
summary alone:

- `logits`, the policy over the 422 actions: a claim from its route's token, the colour paid and
  the global token; a face-up draw from its slot's token; the blind draw and the ticket draw from
  the global token; a keep mask as the sum of its tickets' keep scores plus a term for how many
  are kept. Illegal actions are masked out.
- `value`: the expected utility per seat, +1 first place .. -1 last; `score`: the final score per
  seat / 100. With `outcome_heads="tokens"` they pool every token by attention (`OutcomeHead`),
  with `"global"` they read the global token.
- `belief`: per seat and ticket, the logit that the seat holds the ticket.
- Auxiliary forecasts that shape the trunk in training: `owner`, who holds each route at the end
  (nobody or a seat); `ticket`, each ticket's outcome for the observer (not held, completed,
  failed); `hand`, each seat's cards by type / 6; `path`, per seat and ticket the trains the seat
  still needs to join the ticket's cities / 45 and whether it still can; `trail` (`trail_head`),
  per seat whether it holds the longest trail at the end and the trail's length then / 30.
- `adv` (`adv_head`), the critic of the policy-gradient phases: the expected reward of the game
  for the seat to act, in hundreds of points.
- `move` (`move_head`): per action, a forecast of the margin over the best other seat that the
  seat to act ends the game with after that move, in hundreds of points.

Seats are relative to the observer everywhere: 0 is the observer, 1 the next seat to act, and so on.
`NetConfig` is the size; the observation does not depend on it.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEATS = 5
EVENTS = 32
OBS_KEYS = ("routes", "cities", "tickets", "players", "global", "slots", "offer", "events", "legal")
# the published network, which comes with the package: `load()` reads it, the commands call it `net`
PUBLISHED = Path(__file__).resolve().parent / "weights" / "deepttr.pt"


@dataclass
class NetConfig:
    """The size and the optional heads of a network; the defaults are the published network's."""

    width: int = 256
    depth: int = 6
    heads: int = 4
    mlp_ratio: int = 4
    # what the value and score heads read: "tokens" - every final token, pooled by attention
    # (`OutcomeHead`); "global" - the global token alone
    outcome_heads: str = "tokens"
    # per seat: whether it holds the longest trail at the end, and the trail's length then
    trail_head: bool = True
    # per action: the forecast of the margin the seat to act ends with after it
    move_head: bool = True
    # the critic: the expected reward of the game for the seat to act
    adv_head: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


# relation types between tokens (the attention bias)
REL_NONE, REL_SELF, REL_ROUTE_SHARE, REL_ROUTE_TWIN, REL_ROUTE_CITY, REL_TICKET_CITY = range(6)
REL_TICKET_ROUTE, REL_TICKET_TICKET = 6, 7
REL_CITY_DIST = 8  # + distance in hops, 1..8 (8 = 8 or more)
N_REL = REL_CITY_DIST + 9


def token_layout(dims: dict) -> dict:
    """Where each kind of token starts in the sequence, and the sequence's length (`total`)."""
    R, C, T = dims["n_routes"], dims["n_cities"], dims["n_tickets"]
    o = {"routes": 0, "cities": R, "tickets": R + C, "players": R + C + T}
    o["global"] = o["players"] + SEATS
    o["slots"] = o["global"] + 1
    o["events"] = o["slots"] + 5
    o["total"] = o["events"] + EVENTS
    return o


def relation_matrix(dims: dict) -> np.ndarray:
    """[tokens, tokens] relation types from the map (static: the same for every position)."""
    R, C, T = dims["n_routes"], dims["n_cities"], dims["n_tickets"]
    ra, rb = dims["route_a"], dims["route_b"]
    ta, tb = dims["ticket_a"], dims["ticket_b"]
    pair = dims["route_pair"]
    lay = token_layout(dims)
    N = lay["total"]
    rel = np.full((N, N), REL_NONE, dtype=np.int64)
    np.fill_diagonal(rel, REL_SELF)
    r0, c0, t0 = lay["routes"], lay["cities"], lay["tickets"]
    ends = [{ra[r], rb[r]} for r in range(R)]
    for i in range(R):
        for j in range(R):
            if i == j:
                continue
            if pair[i] == pair[j]:
                rel[r0 + i, r0 + j] = REL_ROUTE_TWIN
            elif ends[i] & ends[j]:
                rel[r0 + i, r0 + j] = REL_ROUTE_SHARE
        for c in ends[i]:
            rel[r0 + i, c0 + c] = rel[c0 + c, r0 + i] = REL_ROUTE_CITY
    # city distances in hops
    adj = [set() for _ in range(C)]
    for r in range(R):
        adj[ra[r]].add(rb[r])
        adj[rb[r]].add(ra[r])
    for s in range(C):
        dist = [-1] * C
        dist[s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = dist[x] + 1
                    q.append(y)
        for t in range(C):
            if t != s and dist[t] > 0:
                rel[c0 + s, c0 + t] = REL_CITY_DIST + min(dist[t], 8)
    tends = [{ta[t], tb[t]} for t in range(T)]
    for t in range(T):
        for c in tends[t]:
            rel[t0 + t, c0 + c] = rel[c0 + c, t0 + t] = REL_TICKET_CITY
        for r in range(R):
            if ends[r] & tends[t]:
                rel[t0 + t, r0 + r] = rel[r0 + r, t0 + t] = REL_TICKET_ROUTE
        for u in range(T):
            if u != t and tends[t] & tends[u]:
                rel[t0 + t, t0 + u] = REL_TICKET_TICKET
    return rel


class Block(nn.Module):
    """A pre-norm transformer layer whose attention adds `bias` to the scores of every pair of
    tokens. `bias` is [1, heads, tokens, tokens padded to a multiple of 8] and is sliced back to the
    tokens here, after the cast: PyTorch's fused attention kernel on CUDA takes an additive mask
    only in that memory layout, and falls back to an unfused path otherwise - about half as fast,
    and rounding differently."""

    def __init__(self, d: int, heads: int, ratio: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ratio * d)
        self.fc2 = nn.Linear(ratio * d, d)

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype)[..., :N])
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class OutcomeHead(nn.Module):
    """A forecast of the game's end that reads every final token: learned queries pool the tokens
    by attention, then an MLP reads the pools and the global token. The outcome of a game is spread
    over the board - who holds which route, which tickets are still open - and the global token
    alone does not carry it."""

    def __init__(self, d: int, queries: int = 4, outputs: int = SEATS) -> None:
        super().__init__()
        self.q = nn.Parameter(torch.randn(queries, d) / math.sqrt(d))
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d * (queries + 1), 2 * d), nn.GELU(), nn.Linear(2 * d, outputs))

    def forward(self, x: torch.Tensor, h_g: torch.Tensor) -> torch.Tensor:
        att = torch.softmax(torch.einsum("qd,bnd->bqn", self.q, self.k(x)) / math.sqrt(x.shape[-1]), -1)
        pooled = torch.einsum("bqn,bnd->bqd", att, self.v(x)).flatten(1)
        return self.mlp(torch.cat([pooled, h_g], -1))


class DeepTTRNet(nn.Module):
    """`dims`: the sizes of an observation and the static facts of the map (`deepttr._engine.dims()`;
    a checkpoint carries the ones its network was built from)."""

    def __init__(self, dims: dict, cfg: NetConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or NetConfig()
        self.cfg = cfg
        self.dims = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dims.items()}
        d = cfg.width
        R, C, T = dims["n_routes"], dims["n_cities"], dims["n_tickets"]
        self.R, self.C, self.T, self.A = R, C, T, dims["n_actions"]
        self.n_claim = dims["n_claim"]
        self.lay = token_layout(dims)
        self.register_buffer("relmat", torch.from_numpy(relation_matrix(dims)), persistent=False)
        self.register_buffer("claim_route", torch.tensor(dims["claim_route"], dtype=torch.long), persistent=False)
        self.register_buffer("claim_color", torch.tensor(dims["claim_color"], dtype=torch.long), persistent=False)
        # the seven keep masks of a ticket choice: mask m keeps the tickets on offer of its set bits
        masks = torch.tensor([[(m >> i) & 1 for i in range(3)] for m in range(1, 8)], dtype=torch.float32)
        self.register_buffer("keep_bits", masks, persistent=False)  # [7, 3]
        self.register_buffer("keep_count", masks.sum(1).long(), persistent=False)

        # tokens: a linear layer over each entity's features plus an embedding of which entity it is
        self.route_in = nn.Linear(dims["route_f"], d)
        self.route_id = nn.Embedding(R, d)
        self.city_in = nn.Linear(dims["city_f"], d)
        self.city_id = nn.Embedding(C, d)
        self.ticket_in = nn.Linear(dims["ticket_f"], d)
        self.ticket_id = nn.Embedding(T, d)
        self.player_in = nn.Linear(dims["player_f"], d)
        self.seat_id = nn.Embedding(SEATS, d)
        self.global_in = nn.Linear(dims["global_f"], d)
        self.slot_card = nn.Embedding(10, d)
        self.slot_pos = nn.Embedding(5, d)
        self.ev_kind = nn.Embedding(10, d)
        self.ev_seat = nn.Embedding(SEATS + 1, d)
        self.ev_card = nn.Embedding(10, d)
        self.ev_route = nn.Embedding(R + 1, d)
        self.ev_count = nn.Embedding(8, d)
        self.ev_pos = nn.Embedding(EVENTS, d)

        # the trunk
        self.rel_bias = nn.Embedding(N_REL, cfg.heads)
        nn.init.zeros_(self.rel_bias.weight)
        self.blocks = nn.ModuleList(Block(d, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.ln_f = nn.LayerNorm(d)

        # the policy
        self.claim_color_emb = nn.Embedding(8, d)
        self.claim_ctx = nn.Linear(d, d)
        self.claim_out = nn.Sequential(nn.GELU(), nn.Linear(d, 1))
        self.slot_head = nn.Linear(d, 1)
        self.global_head = nn.Linear(d, 2)  # blind draw, ticket draw
        self.keep_ticket = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.keep_size = nn.Embedding(4, 1)

        # the outcome of the game
        if cfg.outcome_heads == "tokens":
            self.value_head = OutcomeHead(d)
            self.score_head = OutcomeHead(d)
        else:
            self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, SEATS))
            self.score_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, SEATS))

        # the belief and the auxiliary forecasts
        self.belief_p = nn.Linear(d, d)
        self.belief_t = nn.Linear(d, d)
        self.owner_head = nn.Linear(d, SEATS + 1)
        self.ticket_head = nn.Linear(d, 3)
        self.hand_head = nn.Linear(d, dims.get("card_types", 9))
        self.path_head = nn.Linear(d, 2 * SEATS)
        if cfg.trail_head:
            self.trail_head = nn.Linear(d, 2)
        if cfg.adv_head:
            # starts at 0: an untrained critic then changes nothing in the policy gradient
            self.adv_head = OutcomeHead(d, outputs=1) if cfg.outcome_heads == "tokens" else nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            last = self.adv_head.mlp[-1] if cfg.outcome_heads == "tokens" else self.adv_head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        if cfg.move_head:
            # reads the moves as the policy reads them, by layers of its own; starts at 0, so an
            # untrained forecast ranks no move above another
            self.move_claim = nn.Sequential(nn.GELU(), nn.Linear(d, 1))
            self.move_slot = nn.Linear(d, 1)
            self.move_global = nn.Linear(d, 2)  # blind draw, ticket draw
            self.move_keep = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
            self.move_keep_size = nn.Embedding(4, 1)
            for last in (self.move_claim[-1], self.move_slot, self.move_global, self.move_keep[-1]):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            nn.init.zeros_(self.move_keep_size.weight)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, obs: dict) -> dict:
        lay = self.lay
        R, C, T = self.R, self.C, self.T
        dev = self.route_id.weight.device
        B = obs["routes"].shape[0]
        ar = lambda n: torch.arange(n, device=dev)  # noqa: E731

        # the 209 tokens
        routes = self.route_in(obs["routes"]) + self.route_id(ar(R))
        cities = self.city_in(obs["cities"]) + self.city_id(ar(C))
        tickets = self.ticket_in(obs["tickets"]) + self.ticket_id(ar(T))
        players = self.player_in(obs["players"]) + self.seat_id(ar(SEATS))
        glob = self.global_in(obs["global"]).unsqueeze(1)
        slots = self.slot_card(obs["slots"].long()) + self.slot_pos(ar(5))
        ev = obs["events"].long()
        events = (
            self.ev_kind(ev[..., 0]) + self.ev_seat(ev[..., 1]) + self.ev_card(ev[..., 2])
            + self.ev_route(ev[..., 3]) + self.ev_count(ev[..., 4].clamp(0, 7)) + self.ev_pos(ar(EVENTS))
        )
        x = torch.cat([routes, cities, tickets, players, glob, slots, events], dim=1)

        # the trunk: every layer attends over all the tokens, biased by their relation on the map
        rb = self.rel_bias(self.relmat).permute(2, 0, 1)  # [heads, tokens, tokens]
        N = rb.shape[-1]
        bias = rb.new_zeros(1, rb.shape[0], N, (N + 7) // 8 * 8)  # padded for the fused kernel (`Block`)
        bias[..., :N] = rb
        for blk in self.blocks:
            x = blk(x, bias)
        x = self.ln_f(x)

        h_r = x[:, lay["routes"]:lay["routes"] + R]
        h_t = x[:, lay["tickets"]:lay["tickets"] + T]
        h_p = x[:, lay["players"]:lay["players"] + SEATS]
        h_g = x[:, lay["global"]]
        h_s = x[:, lay["slots"]:lay["slots"] + 5]

        # the policy, in action-id order: claims, face-up slots, the blind draw, the ticket draw, keep masks
        claim = h_r[:, self.claim_route] + self.claim_color_emb(self.claim_color) + self.claim_ctx(h_g).unsqueeze(1)
        claim_logits = self.claim_out(claim).squeeze(-1)  # [B, n_claim]
        slot_logits = self.slot_head(h_s).squeeze(-1)  # [B, 5]
        g2 = self.global_head(h_g)  # [B, 2]
        offer = obs["offer"].long()  # [B, 3], T = none
        h_t_pad = torch.cat([h_t, torch.zeros_like(h_t[:, :1])], dim=1)
        h_off = torch.gather(h_t_pad, 1, offer.unsqueeze(-1).expand(-1, -1, h_t.shape[-1]))  # [B, 3, d]
        s_off = self.keep_ticket(torch.cat([h_off, h_g.unsqueeze(1).expand_as(h_off)], -1)).squeeze(-1)  # [B, 3]
        keep_logits = s_off @ self.keep_bits.t().to(s_off.dtype) + self.keep_size(self.keep_count).squeeze(-1)  # [B, 7]
        logits = torch.cat([claim_logits, slot_logits, g2[:, :1], g2[:, 1:], keep_logits], dim=1).float()
        logits = logits.masked_fill(obs["legal"] == 0, -1e9)

        belief = torch.einsum("bpd,btd->bpt", self.belief_p(h_p), self.belief_t(h_t)) / math.sqrt(h_t.shape[-1])
        path = self.path_head(h_t).float().view(B, T, SEATS, 2).transpose(1, 2)  # [B, seats, T, (cost, can)]
        if self.cfg.outcome_heads == "tokens":
            value, score = self.value_head(x, h_g), self.score_head(x, h_g)
        else:
            value, score = self.value_head(h_g), self.score_head(h_g)
        out = {
            "logits": logits,
            "value": torch.tanh(value.float()),
            "score": score.float(),
            "belief": belief.float(),
            "owner": self.owner_head(h_r).float(),
            "ticket": self.ticket_head(h_t).float(),
            "hand": F.softplus(self.hand_head(h_p).float()),
            "path": path,
        }
        if self.cfg.trail_head:  # [B, seats, (holds the longest at the end, its length / 30)]
            out["trail"] = self.trail_head(h_p).float()
        if self.cfg.adv_head:  # [B], hundreds of points
            adv = self.adv_head(x, h_g) if self.cfg.outcome_heads == "tokens" else self.adv_head(h_g)
            out["adv"] = adv.float().squeeze(-1)
        if self.cfg.move_head:  # [B, n_actions], hundreds of points; an illegal move's is meaningless
            # detached: the forecast's loss trains its own layers and leaves the trunk to the policy
            m_claim, m_s, m_off, m_gl = claim.detach(), h_s.detach(), h_off.detach(), h_g.detach()
            m_keep = self.move_keep(torch.cat([m_off, m_gl.unsqueeze(1).expand_as(m_off)], -1)).squeeze(-1)  # [B, 3]
            m_keep = m_keep @ self.keep_bits.t().to(m_keep.dtype) + self.move_keep_size(self.keep_count).squeeze(-1)  # [B, 7]
            m_g = self.move_global(m_gl)
            out["move"] = torch.cat([self.move_claim(m_claim).squeeze(-1), self.move_slot(m_s).squeeze(-1), m_g[:, :1], m_g[:, 1:],
                                     m_keep], dim=1).float()
        return out


def to_device(obs: dict, device: torch.device) -> dict:
    """The engine's observation arrays (numpy) -> tensors on `device`."""
    return {k: torch.from_numpy(np.ascontiguousarray(obs[k])).to(device, non_blocking=True) for k in OBS_KEYS}


def save(path, net: DeepTTRNet, **extra) -> None:
    """A checkpoint: the network's config, the map facts it was built from, its weights, and
    `extra` (plain values, e.g. `step=...`)."""
    torch.save({"config": net.cfg.as_dict(), "dims": net.dims, "state": net.state_dict(), **extra}, path)


def load(path=None, device: str | torch.device = "cpu") -> tuple[DeepTTRNet, dict]:
    """A checkpoint -> (its network on `device`, in evaluation mode; the checkpoint's other fields).
    Without a `path`, the published network that comes with the package (`PUBLISHED`). Only
    tensors and plain values are read from the file (`weights_only`)."""
    path = PUBLISHED if path is None else path
    blob = torch.load(path, map_location="cpu", weights_only=True)
    known = NetConfig.__dataclass_fields__
    extra = {k: v for k, v in blob["config"].items() if k not in known and v}
    if extra:  # an input or a head this code does not have, switched on in the checkpoint
        raise ValueError(f"{path}: the checkpoint's network needs what this code has not: {extra}")
    # a checkpoint names the heads its network has: one it does not name is absent
    absent = {"outcome_heads": "global", "trail_head": False, "move_head": False, "adv_head": False}
    config = {**absent, **{k: v for k, v in blob["config"].items() if k in known}}
    net = DeepTTRNet(blob["dims"], NetConfig(**config))
    net.load_state_dict(blob["state"])
    return net.to(device).eval(), {k: v for k, v in blob.items() if k not in ("config", "dims", "state")}


def grow(net: DeepTTRNet, **heads) -> DeepTTRNet:
    """The network with the heads `heads` asks for - `trail_head`, `adv_head` or `move_head` True, or
    `outcome_heads` of a kind - and every weight of `net` that it still has. A new head starts as
    `DeepTTRNet` builds it (the critic and the move forecast at zero), and so do the value, score
    and critic heads when the kind of outcome head changes. A head is never taken away: `net`
    itself when it has what is asked. Training adds the heads a phase needs this way."""
    known = ("outcome_heads", "trail_head", "adv_head", "move_head")
    if set(heads) - set(known):
        raise ValueError(f"grow: heads among {known}, not {sorted(set(heads) - set(known))}")
    old = net.cfg.as_dict()
    want = {**old, **{k: (v if k == "outcome_heads" else bool(old[k] or v)) for k, v in heads.items()}}
    if want == old:
        return net
    new_heads = [p for k, p in (("trail_head", "trail_head."), ("adv_head", "adv_head."), ("move_head", "move_")) if want[k] and not old[k]]
    state = net.state_dict()
    if want["outcome_heads"] != old["outcome_heads"]:
        new_heads += ["value_head.", "score_head.", "adv_head."]
        state = {k: v for k, v in state.items() if not k.startswith(("value_head.", "score_head.", "adv_head."))}
    new = DeepTTRNet(net.dims, NetConfig(**want)).to(next(net.parameters()).device)
    missing, unexpected = new.load_state_dict(state, strict=False)
    if unexpected or not all(k.startswith(tuple(new_heads)) for k in missing):
        raise RuntimeError(f"grow: the weights do not fit: missing {missing}, unexpected {unexpected}")
    return new.train(net.training)
