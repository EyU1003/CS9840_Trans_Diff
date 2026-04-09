import math
import os
import csv
import json
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import osmnx as ox
import networkx as nx
from itertools import islice

try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False


# ============================================================
# Global config
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "transport_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA_BPR = 0.15
BETA_BPR = 4.0
SEGMENTS = 8

# Diffusion enhancement settings
N_DIFFUSION_SAMPLES = 1000
TOPK_ACTIVE = 1                 # keep top-1 candidate edge only; try 2 if needed
USE_QUANTIZATION = True
USE_TOPK = True

# Globals initialized later
G = None
edge_index = None
edge_attr = None
demand_dict = None
od_pairs = None
PATH_EDGE_INCIDENCE = None
paths = None

BASE_CAPACITY = None
BASE_TIME = None

N_EDGES = None
N_PATHS = None
CANDIDATE_EDGES = None
N_CAND = None
CANDIDATE_SET = None
FIXED_EDGES = None
EDGE_NAMES = None

LEVELS = None
MAX_EXPANSION = None

PATH_EDGE_T = None
BASE_CAP_T = None
BASE_TIME_T = None


# ============================================================
# Network utilities
# ============================================================
def convert_to_simple_digraph(G_multi):
    G_simple = nx.DiGraph()
    for u, v, data in G_multi.edges(data=True):
        length = data.get("length", 1.0)
        if G_simple.has_edge(u, v):
            if length < G_simple[u][v]["length"]:
                G_simple[u][v].update(data)
        else:
            G_simple.add_edge(u, v, **data)
    return G_simple


def build_osm_network(place="Western University, London, Ontario, Canada"):
    print("Downloading OSM network...")

    G = ox.graph_from_place(place, network_type="drive")
    G = nx.convert_node_labels_to_integers(G)
    G = convert_to_simple_digraph(G)

    edge_index = []
    edge_attr = []

    for u, v, data in G.edges(data=True):
        length = float(data.get("length", 50.0))
        speed = data.get("speed_kph", 40.0)
        if isinstance(speed, list):
            speed = speed[0]
        speed = float(speed) if speed is not None else 40.0
        speed_mps = max(speed * 1000.0 / 3600.0, 1.0)

        free_time = length / speed_mps
        capacity = 10.0

        edge_index.append([u, v])
        edge_attr.append([length, free_time, capacity])

    edge_index = np.array(edge_index, dtype=np.int64)
    edge_attr = np.array(edge_attr, dtype=np.float32)

    return G, edge_index, edge_attr


def generate_od_demand(G, num_pairs=20):
    nodes = list(G.nodes())
    demand = {}

    while len(demand) < num_pairs:
        o = np.random.choice(nodes)
        d = np.random.choice(nodes)
        if o != d and (o, d) not in demand:
            demand[(o, d)] = np.random.uniform(50, 200)

    return demand


def k_shortest_paths(G, source, target, k=3, weight="length"):
    try:
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))
    except Exception:
        return []


def build_incidence_matrix(G, edge_index, od_pairs, k_paths=3):
    paths = []

    for (o, d) in od_pairs:
        od_paths = k_shortest_paths(G, o, d, k=k_paths, weight="length")
        paths.extend(od_paths)

    N_paths_local = len(paths)
    N_edges_local = len(edge_index)

    incidence = np.zeros((N_paths_local, N_edges_local), dtype=np.float32)
    edge_dict = {(u, v): i for i, (u, v) in enumerate(edge_index)}

    for i, path in enumerate(paths):
        for j in range(len(path) - 1):
            e = (path[j], path[j + 1])
            if e in edge_dict:
                incidence[i, edge_dict[e]] = 1.0

    return incidence, paths


# ============================================================
# UE solver and utilities
# ============================================================
def edge_travel_time_np(edge_flows: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    ratio = np.maximum(edge_flows / np.maximum(capacities, 1e-6), 0.0)
    return BASE_TIME * (1.0 + ALPHA_BPR * ratio ** BETA_BPR)


def beckmann_objective_np(edge_flows: np.ndarray, capacities: np.ndarray) -> float:
    return float(np.sum(
        BASE_TIME * (
            edge_flows
            + ALPHA_BPR / (BETA_BPR + 1.0)
            * np.power(edge_flows, BETA_BPR + 1.0)
            / np.power(np.maximum(capacities, 1e-6), BETA_BPR)
        )
    ))


def all_or_nothing_assignment_np(path_costs: np.ndarray, demand: float) -> np.ndarray:
    x = np.zeros(N_PATHS, dtype=np.float32)
    x[int(np.argmin(path_costs))] = demand
    return x


def line_search_np(edge_flows: np.ndarray, aon_edge_flows: np.ndarray, capacities: np.ndarray) -> float:
    alphas = np.linspace(0.0, 1.0, 41)
    vals = []
    for a in alphas:
        f = (1.0 - a) * edge_flows + a * aon_edge_flows
        vals.append(beckmann_objective_np(f, capacities))
    return float(alphas[int(np.argmin(vals))])


def solve_ue_no_expansion(demand: float, max_iter: int = 80):
    capacities = BASE_CAPACITY.copy()

    path_flows = np.ones(N_PATHS, dtype=np.float32) * (demand / N_PATHS)
    edge_flows = PATH_EDGE_INCIDENCE.T @ path_flows

    for _ in range(max_iter):
        edge_times = edge_travel_time_np(edge_flows, capacities)
        path_costs = PATH_EDGE_INCIDENCE @ edge_times
        aon_path = all_or_nothing_assignment_np(path_costs, demand)
        aon_edge = PATH_EDGE_INCIDENCE.T @ aon_path
        step = line_search_np(edge_flows, aon_edge, capacities)
        path_flows = (1.0 - step) * path_flows + step * aon_path
        edge_flows = PATH_EDGE_INCIDENCE.T @ path_flows

    edge_times = edge_travel_time_np(edge_flows, capacities)
    return edge_flows, edge_times


def select_candidate_edges(edge_attr, k=10):
    screening_demand = 16.0
    edge_flows, edge_times = solve_ue_no_expansion(screening_demand)

    scores = edge_flows * edge_times
    idx = np.argsort(scores)[-k:]
    return idx.astype(np.int64)


def apply_capacity_additions_np(capacity_add: np.ndarray) -> np.ndarray:
    capacities = BASE_CAPACITY.copy()
    assert capacity_add.shape[0] == N_CAND
    for j, e in enumerate(CANDIDATE_EDGES):
        capacities[e] += float(capacity_add[j])
    return capacities


def solve_ue_frank_wolfe(capacity_add: np.ndarray, demand: float, max_iter: int = 80) -> Dict[str, np.ndarray]:
    capacities = apply_capacity_additions_np(capacity_add)

    path_flows = np.ones(N_PATHS, dtype=np.float32) * (demand / N_PATHS)
    edge_flows = PATH_EDGE_INCIDENCE.T @ path_flows

    for _ in range(max_iter):
        edge_times = edge_travel_time_np(edge_flows, capacities)
        path_costs = PATH_EDGE_INCIDENCE @ edge_times
        aon_path = all_or_nothing_assignment_np(path_costs, demand)
        aon_edge = PATH_EDGE_INCIDENCE.T @ aon_path
        step = line_search_np(edge_flows, aon_edge, capacities)
        path_flows = (1.0 - step) * path_flows + step * aon_path
        edge_flows = PATH_EDGE_INCIDENCE.T @ path_flows

    edge_times = edge_travel_time_np(edge_flows, capacities)
    path_costs = PATH_EDGE_INCIDENCE @ edge_times
    total_travel_time = float(np.sum(edge_flows * edge_times))

    return {
        "capacity_add": capacity_add.copy(),
        "capacities": capacities,
        "path_flows": path_flows,
        "edge_flows": edge_flows,
        "edge_times": edge_times,
        "path_costs": path_costs,
        "total_travel_time": total_travel_time,
        "beckmann": beckmann_objective_np(edge_flows, capacities),
    }


# ============================================================
# Scenario generation
# ============================================================
@dataclass
class Scenario:
    demand: float
    budget: float
    peak_signal: float
    features: np.ndarray


def sample_scenario() -> Scenario:
    demand = np.random.uniform(10.0, 18.0)
    budget = np.random.uniform(2.0, 10.0)
    peak_signal = np.random.uniform(0.0, 1.0)

    demand_scale = demand / 14.0
    budget_signal = budget / 10.0
    features = np.array([demand_scale, budget_signal, peak_signal], dtype=np.float32)

    return Scenario(
        demand=demand,
        budget=budget,
        peak_signal=peak_signal,
        features=features,
    )


def generate_dataset(n: int) -> List[Scenario]:
    return [sample_scenario() for _ in range(n)]


# ============================================================
# MILP baseline
# ============================================================
def segment_marginals(base_time: float, capacity: float, max_flow: float, segments: int) -> List[float]:
    seg_w = max_flow / segments
    out = []
    for k in range(segments):
        flow_mid = (k + 0.5) * seg_w
        marginal = base_time * (1.0 + ALPHA_BPR * (flow_mid / max(capacity, 1e-6)) ** BETA_BPR)
        out.append(float(marginal))
    return out


def solve_milp_baseline(scenario: Scenario) -> np.ndarray:
    if not PULP_AVAILABLE:
        raise RuntimeError("PuLP is not installed. Run: pip install pulp")

    demand = float(scenario.demand)
    budget = float(scenario.budget)
    max_flow = demand
    seg_w = max_flow / SEGMENTS

    prob = pulp.LpProblem("MILP_TNDP_Baseline", pulp.LpMinimize)

    x = [pulp.LpVariable(f"x_{p}", lowBound=0.0) for p in range(N_PATHS)]
    prob += pulp.lpSum(x) == demand

    edge_flow_expr = []
    for e in range(N_EDGES):
        expr = pulp.lpSum(float(PATH_EDGE_INCIDENCE[p, e]) * x[p] for p in range(N_PATHS))
        edge_flow_expr.append(expr)

    sel = {}
    z_expr = {}
    for j in range(N_CAND):
        feasible_levels = [float(lv) for lv in LEVELS if lv <= budget + 1e-9]
        bins = [
            pulp.LpVariable(f"sel_{j}_{i}", lowBound=0, upBound=1, cat="Binary")
            for i in range(len(feasible_levels))
        ]
        sel[j] = (feasible_levels, bins)

        prob += pulp.lpSum(bins) == 1
        z_expr[j] = pulp.lpSum(feasible_levels[i] * bins[i] for i in range(len(feasible_levels)))

    prob += pulp.lpSum(z_expr[j] for j in range(N_CAND)) <= budget

    obj_terms = []

    for e in FIXED_EDGES:
        y = [
            pulp.LpVariable(f"y_fix_{e}_{k}", lowBound=0.0, upBound=seg_w)
            for k in range(SEGMENTS)
        ]
        prob += pulp.lpSum(y) == edge_flow_expr[e]

        marginals = segment_marginals(BASE_TIME[e], BASE_CAPACITY[e], max_flow, SEGMENTS)
        for k in range(SEGMENTS):
            obj_terms.append(marginals[k] * y[k])

    for loc_j, e in enumerate(CANDIDATE_EDGES):
        feasible_levels, bins = sel[loc_j]
        y_all = []

        for i, lv in enumerate(feasible_levels):
            cap = float(BASE_CAPACITY[e] + lv)
            marginals = segment_marginals(BASE_TIME[e], cap, max_flow, SEGMENTS)

            y_i = [
                pulp.LpVariable(f"y_cand_{e}_{i}_{k}", lowBound=0.0, upBound=seg_w)
                for k in range(SEGMENTS)
            ]
            for k in range(SEGMENTS):
                prob += y_i[k] <= seg_w * bins[i]
                obj_terms.append(marginals[k] * y_i[k])

            y_all.extend(y_i)

        prob += pulp.lpSum(y_all) == edge_flow_expr[e]

    INVENT_PENALTY = 0.000
    prob += pulp.lpSum(obj_terms) + INVENT_PENALTY * pulp.lpSum(z_expr[j] for j in range(N_CAND))

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    if status not in [1, pulp.LpStatusOptimal, pulp.LpStatusNotSolved]:
        raise RuntimeError(f"MILP solver status: {pulp.LpStatus[status]}")

    z_sol = np.array([
        float(pulp.value(z_expr[j])) if pulp.value(z_expr[j]) is not None else 0.0
        for j in range(N_CAND)
    ], dtype=np.float32)

    return z_sol


# ============================================================
# Differentiable UE layer for training
# ============================================================
def apply_capacity_additions_t(capacity_add: torch.Tensor) -> torch.Tensor:
    batch = capacity_add.shape[0]
    capacities = BASE_CAP_T.unsqueeze(0).repeat(batch, 1)
    assert capacity_add.shape[1] == N_CAND
    for j, e in enumerate(CANDIDATE_EDGES):
        capacities[:, e] += capacity_add[:, j]
    return capacities


def edge_travel_time_t(edge_flows: torch.Tensor, capacities: torch.Tensor) -> torch.Tensor:
    ratio = edge_flows / torch.clamp(capacities, min=1e-6)
    return BASE_TIME_T.unsqueeze(0) * (1.0 + ALPHA_BPR * ratio.pow(BETA_BPR))


def soft_assignment_t(path_costs: torch.Tensor, demand: torch.Tensor, tau: float = 0.35) -> torch.Tensor:
    weights = torch.softmax(-path_costs / tau, dim=-1)
    return demand.unsqueeze(-1) * weights


def differentiable_ue_torch(
    capacity_add: torch.Tensor,
    demand: torch.Tensor,
    iters: int = 30,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = capacity_add.shape[0]

    capacities = apply_capacity_additions_t(capacity_add)

    path_flows = torch.ones((batch, N_PATHS), device=DEVICE) / N_PATHS
    path_flows = path_flows * demand.unsqueeze(-1)

    for k in range(iters):
        edge_flows = path_flows @ PATH_EDGE_T
        edge_times = edge_travel_time_t(edge_flows, capacities)
        path_costs = edge_times @ PATH_EDGE_T.T
        target_path_flows = soft_assignment_t(path_costs, demand)

        step = 2.0 / (k + 2.0)
        path_flows = (1.0 - step) * path_flows + step * target_path_flows

    edge_flows = path_flows @ PATH_EDGE_T
    edge_times = edge_travel_time_t(edge_flows, capacities)
    total_travel_time = torch.sum(edge_flows * edge_times, dim=1)

    return path_flows, edge_flows, edge_times, total_travel_time


# ============================================================
# Post-processing for diffusion outputs
# ============================================================
def budget_projection_np(z: np.ndarray, budget: float) -> np.ndarray:
    z = np.maximum(z, 0.0)
    total = float(np.sum(z))
    if total <= budget + 1e-9:
        return z
    return z * (budget / max(total, 1e-9))


def topk_projection_np(z: np.ndarray, k: int = 1) -> np.ndarray:
    if k >= len(z):
        return z.copy()
    out = np.zeros_like(z)
    idx = np.argsort(z)[-k:]
    out[idx] = z[idx]
    return out


def quantize_to_levels_np(z: np.ndarray, levels: np.ndarray) -> np.ndarray:
    # snap each coordinate to nearest allowed discrete level
    idx = np.argmin(np.abs(z[:, None] - levels[None, :]), axis=1)
    return levels[idx].astype(np.float32)


def repair_quantized_budget_np(z: np.ndarray, budget: float, levels: np.ndarray) -> np.ndarray:
    # If quantization caused budget overflow, greedily reduce smallest-benefit entries
    z = z.copy()
    if np.sum(z) <= budget + 1e-9:
        return z

    descending_levels = sorted([float(v) for v in levels], reverse=True)
    lower_map = {}
    for i, lv in enumerate(descending_levels):
        lower_map[lv] = descending_levels[i + 1] if i + 1 < len(descending_levels) else 0.0

    while np.sum(z) > budget + 1e-9:
        positive_idx = np.where(z > 0)[0]
        if len(positive_idx) == 0:
            break

        # reduce the smallest currently active level first
        current_vals = z[positive_idx]
        pick_local = int(np.argmin(current_vals))
        pick = int(positive_idx[pick_local])
        z[pick] = lower_map.get(float(z[pick]), 0.0)

    return z


def postprocess_diffusion_sample(z: np.ndarray, budget: float) -> np.ndarray:
    z = np.maximum(z, 0.0)

    if USE_TOPK:
        z = topk_projection_np(z, k=TOPK_ACTIVE)

    z = budget_projection_np(z, budget)

    if USE_QUANTIZATION:
        z = quantize_to_levels_np(z, LEVELS)
        z = repair_quantized_budget_np(z, budget, LEVELS)

    return z.astype(np.float32)


def sample_best_diffusion_plan(
    model,
    scenario: Scenario,
    n_samples: int = N_DIFFUSION_SAMPLES,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Dict[str, float]], float, float]:
    with torch.no_grad():
        cond = torch.tensor(scenario.features[None, :], dtype=torch.float32, device=DEVICE)
        cond_rep = cond.repeat(n_samples, 1)
        t_sample0 = time.perf_counter()
        raw_samples = model.sample(cond_rep).cpu().numpy()
        sample_seconds = time.perf_counter() - t_sample0

    candidates_info = []
    best_z = None
    best_res = None
    best_ttt = float("inf")
    ue_seconds = 0.0

    for s in range(n_samples):
        t_ue0 = time.perf_counter()
        z = postprocess_diffusion_sample(raw_samples[s], scenario.budget)
        res = solve_ue_frank_wolfe(z, scenario.demand)
        ue_seconds += time.perf_counter() - t_ue0
        ttt = float(res["total_travel_time"])

        candidates_info.append({
            "sample_id": s,
            "ttt": ttt,
            "total_expansion": float(np.sum(z)),
            "active_edges": int(np.sum(z > 1e-6)),
        })

        if ttt < best_ttt:
            best_ttt = ttt
            best_z = z.copy()
            best_res = res

    return best_z, best_res, candidates_info, sample_seconds, ue_seconds


# ============================================================
# Dataset creation
# ============================================================
def build_labeled_dataset(scenarios: List[Scenario]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    X, Y, TTT = [], [], []
    milp_solve_s = 0.0
    for sc in scenarios:
        t0 = time.perf_counter()
        z = solve_milp_baseline(sc)
        milp_solve_s += time.perf_counter() - t0
        res = solve_ue_frank_wolfe(z, sc.demand)
        X.append(sc.features)
        Y.append(z)
        TTT.append(res["total_travel_time"])
    return np.stack(X), np.stack(Y), np.array(TTT, dtype=np.float32), milp_solve_s


# ============================================================
# Diffusion model
# ============================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-torch.linspace(0.0, math.log(1000.0), half, device=t.device))
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Denoiser(nn.Module):
    def __init__(self, feature_dim: int = 3, time_dim: int = 32, hidden_dim: int = 128, out_dim: int = None):
        super().__init__()
        if out_dim is None:
            out_dim = N_CAND

        self.time_net = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(feature_dim + out_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t_scaled: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_net(t_scaled)
        h = torch.cat([x_t, cond, t_emb], dim=-1)
        return self.net(h)


class DiffusionCapacityModel(nn.Module):
    def __init__(self, timesteps: int = 40):
        super().__init__()
        self.timesteps = timesteps
        self.denoiser = Denoiser(out_dim=N_CAND)

        betas = torch.linspace(1e-4, 2e-2, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, x0: torch.Tensor, t_idx: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t_idx].unsqueeze(-1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    def predict_noise(self, x_t: torch.Tensor, cond: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        t_scaled = t_idx.float() / float(self.timesteps - 1)
        return self.denoiser(x_t, cond, t_scaled)

    def x0_from_eps(self, x_t: torch.Tensor, t_idx: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t_idx].unsqueeze(-1)
        return (x_t - torch.sqrt(1.0 - a_bar) * eps) / torch.sqrt(a_bar)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor) -> torch.Tensor:
        self.eval()
        n = cond.shape[0]
        x = torch.randn((n, N_CAND), device=cond.device)

        for t in reversed(range(self.timesteps)):
            t_idx = torch.full((n,), t, dtype=torch.long, device=cond.device)
            eps = self.predict_noise(x, cond, t_idx)

            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]

            x = (1.0 / torch.sqrt(alpha)) * (
                x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)) * eps
            )

            if t > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)

        return torch.clamp(x, min=0.0, max=float(MAX_EXPANSION.max()))


# ============================================================
# Training
# ============================================================
def make_tensors(X: np.ndarray, Y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(X, dtype=torch.float32, device=DEVICE),
        torch.tensor(Y, dtype=torch.float32, device=DEVICE),
    )


def budget_projection_t(z: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
    total = torch.sum(z, dim=1, keepdim=True)
    scale = torch.minimum(
        torch.ones_like(total),
        budget.unsqueeze(1) / torch.clamp(total, min=1e-6),
    )
    return z * scale


def train_diffusion_model(
    model: DiffusionCapacityModel,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    epochs: int = 220,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    X_t, Y_t = make_tensors(X_train, Y_train)
    n = X_t.shape[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(DEVICE)
    model.train()

    history = []

    for epoch in range(epochs):
        perm = torch.randperm(n, device=DEVICE)

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_sup = 0.0
        epoch_phys = 0.0
        epoch_sparse = 0.0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            cond = X_t[idx]
            x0 = Y_t[idx]

            t_idx = torch.randint(0, model.timesteps, (cond.shape[0],), device=DEVICE)
            noise = torch.randn_like(x0)

            x_t = model.q_sample(x0, t_idx, noise)
            pred_noise = model.predict_noise(x_t, cond, t_idx)

            recon_loss = F.mse_loss(pred_noise, noise)

            x0_hat = model.x0_from_eps(x_t, t_idx, pred_noise)
            x0_hat = torch.clamp(x0_hat, min=0.0, max=float(MAX_EXPANSION.max()))

            budget = cond[:, 1] * 10.0
            x0_hat = budget_projection_t(x0_hat, budget)

            sup_loss = F.mse_loss(x0_hat, x0)

            demand = cond[:, 0] * 14.0
            _, _, _, ttt = differentiable_ue_torch(x0_hat, demand, iters=28)
            physics_loss = torch.mean(ttt) / 40.0

            # encourage concentration / sparse expansion
            sparse_loss = torch.mean(torch.sum(x0_hat, dim=1)) / 10.0

            loss = recon_loss + 0.8 * sup_loss + 0.25 * physics_loss + 0.01 * sparse_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            bsz = cond.shape[0]
            epoch_loss += loss.item() * bsz
            epoch_recon += recon_loss.item() * bsz
            epoch_sup += sup_loss.item() * bsz
            epoch_phys += physics_loss.item() * bsz
            epoch_sparse += sparse_loss.item() * bsz

        row = {
            "epoch": epoch + 1,
            "loss": epoch_loss / n,
            "recon_loss": epoch_recon / n,
            "sup_loss": epoch_sup / n,
            "physics_loss": epoch_phys / n,
            "sparse_loss": epoch_sparse / n,
        }
        history.append(row)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"[Train] Epoch {epoch + 1:03d} | "
                f"Loss={row['loss']:.4f} | "
                f"Recon={row['recon_loss']:.4f} | "
                f"Sup={row['sup_loss']:.4f} | "
                f"Phys={row['physics_loss']:.4f} | "
                f"Sparse={row['sparse_loss']:.4f}"
            )

    with open(os.path.join(OUT_DIR, "training_history.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "loss", "recon_loss", "sup_loss", "physics_loss", "sparse_loss"],
        )
        writer.writeheader()
        writer.writerows(history)


# ============================================================
# Evaluation
# ============================================================
def evaluate_baseline(test_scenarios: List[Scenario]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    rows = []
    ttts = []
    zs = []
    feasible = []
    milp_solve_s = 0.0

    for i, sc in enumerate(test_scenarios):
        t0 = time.perf_counter()
        z = solve_milp_baseline(sc)
        milp_solve_s += time.perf_counter() - t0
        res = solve_ue_frank_wolfe(z, sc.demand)

        row = {
            "scenario_id": i,
            "method": "MILP",
            "demand": sc.demand,
            "budget": sc.budget,
            "peak_signal": sc.peak_signal,
            "ttt": float(res["total_travel_time"]),
        }
        for j, e in enumerate(CANDIDATE_EDGES):
            row[f"z_edge_{e}"] = float(z[j])
        rows.append(row)

        ttts.append(res["total_travel_time"])
        zs.append(z)
        feasible.append(float(z.sum() <= sc.budget + 1e-6))

    zs = np.stack(zs)
    metrics = {
        "mean_ttt": float(np.mean(ttts)),
        "std_ttt": float(np.std(ttts)),
        "budget_feasibility": float(np.mean(feasible)),
        "mean_total_expansion": float(np.mean(np.sum(zs, axis=1))),
        "nonzero_rate": float(np.mean(np.sum(zs, axis=1) > 1e-6)),
        "milp_solve_seconds": float(milp_solve_s),
    }
    for j, e in enumerate(CANDIDATE_EDGES):
        metrics[f"mean_z_edge_{e}"] = float(np.mean(zs[:, j]))
    return metrics, rows


@torch.no_grad()
def evaluate_diffusion(
    model: DiffusionCapacityModel,
    test_scenarios: List[Scenario],
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    rows = []
    ttts = []
    zs = []
    feasible = []
    diffusion_sample_s = 0.0
    diffusion_ue_s = 0.0

    for i, sc in enumerate(test_scenarios):
        z_best, res_best, cand_info, sample_s, ue_s = sample_best_diffusion_plan(
            model, sc, n_samples=N_DIFFUSION_SAMPLES
        )
        diffusion_sample_s += sample_s
        diffusion_ue_s += ue_s

        row = {
            "scenario_id": i,
            "method": "DiffusionBestOfN",
            "demand": sc.demand,
            "budget": sc.budget,
            "peak_signal": sc.peak_signal,
            "ttt": float(res_best["total_travel_time"]),
            "num_samples": N_DIFFUSION_SAMPLES,
        }

        for j, e in enumerate(CANDIDATE_EDGES):
            row[f"z_edge_{e}"] = float(z_best[j])

        rows.append(row)

        ttts.append(res_best["total_travel_time"])
        zs.append(z_best)
        feasible.append(float(np.sum(z_best) <= sc.budget + 1e-6))

    zs = np.stack(zs)
    metrics = {
        "mean_ttt": float(np.mean(ttts)),
        "std_ttt": float(np.std(ttts)),
        "budget_feasibility": float(np.mean(feasible)),
        "mean_total_expansion": float(np.mean(np.sum(zs, axis=1))),
        "nonzero_rate": float(np.mean(np.sum(zs, axis=1) > 1e-6)),
        "diffusion_model_sample_seconds": float(diffusion_sample_s),
        "diffusion_ue_eval_seconds": float(diffusion_ue_s),
        "diffusion_solve_seconds": float(diffusion_sample_s + diffusion_ue_s),
    }

    for j, e in enumerate(CANDIDATE_EDGES):
        metrics[f"mean_z_edge_{e}"] = float(np.mean(zs[:, j]))

    return metrics, rows


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for k, v in metrics.items():
        if k.endswith("_seconds"):
            print(f"{k:32s}: {v:.3f} s")
        else:
            print(f"{k:32s}: {v:.4f}")


def write_rows_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_files(
    baseline_metrics: Dict[str, float],
    diffusion_metrics: Dict[str, float],
    train_milp_solve_seconds: float,
    n_train: int,
) -> None:
    improvement = 100.0 * (
        baseline_metrics["mean_ttt"] - diffusion_metrics["mean_ttt"]
    ) / baseline_metrics["mean_ttt"]

    summary = {
        "training": {
            "n_scenarios": n_train,
            "milp_solve_seconds": float(train_milp_solve_seconds),
        },
        "baseline": baseline_metrics,
        "diffusion": diffusion_metrics,
        "relative_improvement_percent": improvement,
        "diffusion_mode": f"best_of_{N_DIFFUSION_SAMPLES}_samples_topk_{TOPK_ACTIVE}_quantized",
    }

    with open(os.path.join(OUT_DIR, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    latex = f"""
\\begin{{table}}[t]
\\centering
\\begin{{tabular}}{{lccccc}}
\\toprule
Method & Mean TTT & Std TTT & Budget Feas. & Mean Total Expansion & Nonzero Rate \\\\
\\midrule
MILP Baseline & {baseline_metrics['mean_ttt']:.3f} & {baseline_metrics['std_ttt']:.3f} & {baseline_metrics['budget_feasibility']:.3f} & {baseline_metrics['mean_total_expansion']:.3f} & {baseline_metrics['nonzero_rate']:.3f} \\\\
Diffusion Best-of-{N_DIFFUSION_SAMPLES} & {diffusion_metrics['mean_ttt']:.3f} & {diffusion_metrics['std_ttt']:.3f} & {diffusion_metrics['budget_feasibility']:.3f} & {diffusion_metrics['mean_total_expansion']:.3f} & {diffusion_metrics['nonzero_rate']:.3f} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Comparison of the MILP baseline and the diffusion-based model with multi-sample selection under UE evaluation. Relative improvement in mean total travel time is {improvement:.2f}\\%.}}
\\label{{tab:main_results}}
\\end{{table}}
""".strip()

    with open(os.path.join(OUT_DIR, "results_table.tex"), "w") as f:
        f.write(latex)


def run_case_study(model: DiffusionCapacityModel) -> None:
    cases = {
        "Moderate": Scenario(
            demand=14.0,
            budget=8.0,
            peak_signal=0.4,
            features=np.array([1.0, 0.8, 0.4], dtype=np.float32),
        ),
        "HighDemand": Scenario(
            demand=17.0,
            budget=9.0,
            peak_signal=0.8,
            features=np.array([17.0 / 14.0, 0.9, 0.8], dtype=np.float32),
        ),
        "LowBudget": Scenario(
            demand=14.5,
            budget=4.0,
            peak_signal=0.5,
            features=np.array([14.5 / 14.0, 0.4, 0.5], dtype=np.float32),
        ),
    }

    rows = []

    print("\nCASE STUDIES")
    for name, sc in cases.items():
        z_milp = solve_milp_baseline(sc)
        milp_res = solve_ue_frank_wolfe(z_milp, sc.demand)

        z_diff, diff_res, cand_info, _, _ = sample_best_diffusion_plan(model, sc, n_samples=N_DIFFUSION_SAMPLES)

        print("\n" + "-" * 72)
        print(name)
        print("-" * 72)
        print(f"Scenario features [demand_scale, budget_signal, peak_signal]: {sc.features}")
        print(f"Demand: {sc.demand:.3f} | Budget: {sc.budget:.3f}")
        print(f"MILP total travel time:       {milp_res['total_travel_time']:.4f}")
        print(f"Diffusion total travel time:  {diff_res['total_travel_time']:.4f}")
        print(f"Diffusion selection mode:     best of {N_DIFFUSION_SAMPLES} samples")

        milp_str = ", ".join([f"edge_{CANDIDATE_EDGES[j]}={z_milp[j]:.3f}" for j in range(N_CAND)])
        diff_str = ", ".join([f"edge_{CANDIDATE_EDGES[j]}={z_diff[j]:.3f}" for j in range(N_CAND)])

        print(f"MILP capacity additions:      {milp_str}")
        print(f"Diffusion capacity additions: {diff_str}")

        rows.append({
            "case": name,
            "demand": sc.demand,
            "budget": sc.budget,
            "milp_ttt": float(milp_res["total_travel_time"]),
            "diff_ttt": float(diff_res["total_travel_time"]),
            "diff_num_samples": N_DIFFUSION_SAMPLES,
            "milp_total_expansion": float(np.sum(z_milp)),
            "diff_total_expansion": float(np.sum(z_diff)),
        })

    write_rows_csv(os.path.join(OUT_DIR, "case_studies.csv"), rows)


# ============================================================
# Initialization
# ============================================================
def initialize_problem_data(place="Western University, London, Ontario, Canada", num_pairs=20, k_paths=3, k_candidates=10):
    global G, edge_index, edge_attr
    global demand_dict, od_pairs, PATH_EDGE_INCIDENCE, paths
    global BASE_CAPACITY, BASE_TIME
    global N_EDGES, N_PATHS, CANDIDATE_EDGES, N_CAND, CANDIDATE_SET, FIXED_EDGES, EDGE_NAMES
    global LEVELS, MAX_EXPANSION
    global PATH_EDGE_T, BASE_CAP_T, BASE_TIME_T

    G, edge_index, edge_attr = build_osm_network(place=place)

    demand_dict = generate_od_demand(G, num_pairs=num_pairs)
    od_pairs = list(demand_dict.keys())

    PATH_EDGE_INCIDENCE, paths = build_incidence_matrix(G, edge_index, od_pairs, k_paths=k_paths)

    BASE_CAPACITY = edge_attr[:, 2]
    BASE_TIME = edge_attr[:, 1]

    N_EDGES = len(edge_index)
    N_PATHS = PATH_EDGE_INCIDENCE.shape[0]

    CANDIDATE_EDGES = select_candidate_edges(edge_attr, k=k_candidates)
    N_CAND = len(CANDIDATE_EDGES)
    CANDIDATE_SET = set(int(e) for e in CANDIDATE_EDGES)
    FIXED_EDGES = [e for e in range(N_EDGES) if e not in CANDIDATE_SET]

    EDGE_NAMES = [f"e{i}" for i in range(N_EDGES)]

    LEVELS = np.array([0.0, 4.0, 8.0, 12.0, 16.0], dtype=np.float32)
    MAX_EXPANSION = np.full(N_CAND, 16.0, dtype=np.float32)

    PATH_EDGE_T = torch.tensor(PATH_EDGE_INCIDENCE, dtype=torch.float32, device=DEVICE)
    BASE_CAP_T = torch.tensor(BASE_CAPACITY, dtype=torch.float32, device=DEVICE)
    BASE_TIME_T = torch.tensor(BASE_TIME, dtype=torch.float32, device=DEVICE)

    print(f"Initialized network with {N_EDGES} edges, {N_PATHS} paths, {N_CAND} candidate edges.")
    print(f"Candidate edges: {CANDIDATE_EDGES}")
    print(f"Diffusion mode: best_of={N_DIFFUSION_SAMPLES}, topk={TOPK_ACTIVE}, quantized={USE_QUANTIZATION}")


# ============================================================
# Main
# ============================================================
def main() -> None:
    if not PULP_AVAILABLE:
        print("PuLP not found. Install with: pip install pulp")
        return

    print(f"Using device: {DEVICE}")

    initialize_problem_data()

    n_train = 1000
    n_test = 100

    train_scenarios = generate_dataset(n_train)
    test_scenarios = generate_dataset(n_test)

    print("Generating MILP labels for training data...")
    X_train, Y_train, train_ttt, train_milp_solve_s = build_labeled_dataset(train_scenarios)

    print(f"Train label mean TTT (evaluated under UE): {train_ttt.mean():.4f}")
    print(f"Train label nonzero rate: {np.mean(np.sum(Y_train, axis=1) > 1e-6):.4f}")
    print(f"Train label mean total expansion: {np.mean(np.sum(Y_train, axis=1)):.4f}")
    print(f"MILP solve time (train labels, n={n_train}): {train_milp_solve_s:.3f} s")

    model = DiffusionCapacityModel(timesteps=40)
    train_diffusion_model(
        model,
        X_train,
        Y_train,
        epochs=220,
        batch_size=32,
        lr=1e-3,
    )

    baseline_metrics, baseline_rows = evaluate_baseline(test_scenarios)
    diffusion_metrics, diffusion_rows = evaluate_diffusion(model, test_scenarios)

    print_metrics("MILP Baseline (evaluated by UE)", baseline_metrics)
    print_metrics(f"Diffusion Best-of-{N_DIFFUSION_SAMPLES} (evaluated by UE)", diffusion_metrics)
    print(
        f"\nSolve times (test set, n={n_test}): "
        f"MILP {baseline_metrics['milp_solve_seconds']:.3f} s | "
        f"Diffusion total {diffusion_metrics['diffusion_solve_seconds']:.3f} s "
        f"(model.sample {diffusion_metrics['diffusion_model_sample_seconds']:.3f} s, "
        f"UE eval {diffusion_metrics['diffusion_ue_eval_seconds']:.3f} s)"
    )

    improvement = 100.0 * (
        baseline_metrics["mean_ttt"] - diffusion_metrics["mean_ttt"]
    ) / baseline_metrics["mean_ttt"]
    print(f"\nRelative improvement of diffusion over MILP baseline: {improvement:.2f}%")

    write_rows_csv(os.path.join(OUT_DIR, "baseline_test_results.csv"), baseline_rows)
    write_rows_csv(os.path.join(OUT_DIR, "diffusion_test_results.csv"), diffusion_rows)
    save_summary_files(baseline_metrics, diffusion_metrics, train_milp_solve_s, n_train)
    run_case_study(model)

    print("\nSaved files:")
    print(f"- {os.path.join(OUT_DIR, 'training_history.csv')}")
    print(f"- {os.path.join(OUT_DIR, 'baseline_test_results.csv')}")
    print(f"- {os.path.join(OUT_DIR, 'diffusion_test_results.csv')}")
    print(f"- {os.path.join(OUT_DIR, 'summary_metrics.json')}")
    print(f"- {os.path.join(OUT_DIR, 'results_table.tex')}")
    print(f"- {os.path.join(OUT_DIR, 'case_studies.csv')}")


if __name__ == "__main__":
    main()