import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import (
    DEVICE,
    DTYPE,
    FEATURE_DIM,
    HIDDEN_DIM,
    TIME_EMBED_DIM,
    DIFFUSION_STEPS,
    MAX_CAPACITY_ADDITION,
    UE_ITERS,
    ALPHA,
    BETA,
    TEMP,
    REG_WEIGHT,
    BUDGET_WEIGHT,
)
from network import GraphNetwork


class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int = TIME_EMBED_DIM):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B, 1] in [0,1]
        """
        half_dim = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, device=t.device) / max(half_dim - 1, 1)
        )
        args = t * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.emb_dim:
            pad = self.emb_dim - emb.shape[-1]
            emb = F.pad(emb, (0, pad))
        return self.proj(emb)


class DiffusionBackbone(nn.Module):
    """
    Conditional diffusion-style denoiser for capacity additions.
    Input:
        x_t   : noisy capacity vector [B, 2]
        cond  : scenario features     [B, 3]
        t     : normalized timestep   [B, 1]
    Output:
        predicted noise               [B, 2]
    """

    def __init__(self):
        super().__init__()
        self.time_embed = TimeEmbedding(TIME_EMBED_DIM)
        in_dim = 2 + FEATURE_DIM + TIME_EMBED_DIM

        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2),
        )

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = torch.cat([x_t, cond, t_emb], dim=-1)
        return self.net(h)


class DifferentiableUELayer(nn.Module):
    """
    Unrolled smooth UE approximation.
    """

    def __init__(self, net: GraphNetwork):
        super().__init__()
        self.net = net

    def travel_time(self, edge_flows: torch.Tensor, capacities: torch.Tensor) -> torch.Tensor:
        ratio = edge_flows / torch.clamp(capacities, min=1e-3)
        return self.net.t0 * (1.0 + ALPHA * torch.pow(ratio, BETA))

    def beckmann_objective(self, edge_flows: torch.Tensor, capacities: torch.Tensor) -> torch.Tensor:
        term1 = edge_flows
        term2 = (ALPHA / (BETA + 1.0)) * torch.pow(edge_flows, BETA + 1.0) / torch.pow(
            torch.clamp(capacities, min=1e-3), BETA
        )
        return torch.sum(self.net.t0 * (term1 + term2))

    def forward(self, capacities: torch.Tensor, demand: torch.Tensor):
        """
        capacities: [E]
        demand: scalar tensor or shape [1]
        """
        A = self.net.A  # [P, E]
        num_paths = A.shape[0]

        q = demand.squeeze()

        # Initialize path flow uniformly
        x = torch.ones(num_paths, dtype=DTYPE, device=DEVICE) * (q / num_paths)

        for k in range(UE_ITERS):
            edge_flows = x @ A                       # [E]
            edge_times = self.travel_time(edge_flows, capacities)  # [E]
            path_costs = A @ edge_times             # [P]

            weights = torch.softmax(-TEMP * path_costs, dim=0)
            y = q * weights

            gamma = 2.0 / (k + 2.0)
            x = (1.0 - gamma) * x + gamma * y

        final_edge_flows = x @ A
        final_edge_times = self.travel_time(final_edge_flows, capacities)
        final_path_costs = A @ final_edge_times
        total_tt = torch.sum(final_edge_flows * final_edge_times)
        beckmann = self.beckmann_objective(final_edge_flows, capacities)

        return {
            "path_flows": x,
            "edge_flows": final_edge_flows,
            "edge_times": final_edge_times,
            "path_costs": final_path_costs,
            "total_travel_time": total_tt,
            "beckmann": beckmann,
        }


class TransportDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = GraphNetwork()
        self.diffusion = DiffusionBackbone()
        self.ue = DifferentiableUELayer(self.net)

    def sample_capacity_additions(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Reverse diffusion-style sampling.
        cond: [B, 3]
        returns: [B, 2]
        """
        B = cond.shape[0]
        x = torch.randn(B, self.net.num_expandable, dtype=DTYPE, device=DEVICE)

        for step in reversed(range(DIFFUSION_STEPS)):
            t = torch.full((B, 1), step / DIFFUSION_STEPS, dtype=DTYPE, device=DEVICE)
            pred_noise = self.diffusion(x, cond, t)
            x = x - pred_noise / DIFFUSION_STEPS

        cap_add = torch.sigmoid(x) * MAX_CAPACITY_ADDITION
        return cap_add

    def build_capacities(self, cap_add: torch.Tensor) -> torch.Tensor:
        """
        cap_add: [B, 2]
        returns capacities: [B, E]
        """
        B = cap_add.shape[0]
        capacities = self.net.base_capacity.unsqueeze(0).repeat(B, 1)
        capacities[:, self.net.expandable_indices] += cap_add
        return capacities

    def budget_penalty(self, cap_add: torch.Tensor, budget_signal: torch.Tensor) -> torch.Tensor:
        """
        budget_signal in [0,1]
        lower budget => less allowed total expansion
        """
        used = cap_add.sum(dim=-1)
        allowed = 2.0 + 8.0 * budget_signal.squeeze(-1)
        return F.relu(used - allowed).pow(2).mean()

    def forward(self, cond: torch.Tensor):
        """
        cond: [B, 3] = [demand_scale, budget_signal, peak_signal]
        """
        cap_add = self.sample_capacity_additions(cond)      # [B,2]
        capacities = self.build_capacities(cap_add)         # [B,E]

        demand_scale = cond[:, 0:1]
        demand = self.net.base_demand.unsqueeze(0) * demand_scale  # [B,1]

        losses = []
        ttts = []
        ue_outputs = []

        for i in range(cond.shape[0]):
            ue_out = self.ue(capacities[i], demand[i])
            ue_outputs.append(ue_out)
            ttts.append(ue_out["total_travel_time"])
            losses.append(ue_out["total_travel_time"])

        ttt_loss = torch.stack(losses).mean()
        reg_loss = REG_WEIGHT * torch.mean(cap_add.pow(2))
        budget_loss = BUDGET_WEIGHT * self.budget_penalty(cap_add, cond[:, 1:2])

        total_loss = ttt_loss + reg_loss + budget_loss

        return {
            "loss": total_loss,
            "travel_time_loss": ttt_loss,
            "reg_loss": reg_loss,
            "budget_loss": budget_loss,
            "capacity_additions": cap_add,
            "capacities": capacities,
            "demand": demand,
            "ue_outputs": ue_outputs,
        }