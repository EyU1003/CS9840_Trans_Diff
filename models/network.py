import torch
from configs.configs import DEVICE, DTYPE


class GraphNetwork:
    """
    Small toy transportation network.

    Nodes: 0,1,2,3
    Edges:
        e0: 0->1
        e1: 1->3
        e2: 0->2
        e3: 2->3
        e4: 1->2   (expandable)
        e5: 0->3   (expandable direct edge)

    Candidate paths from 0 to 3:
        p0: 0-1-3       [e0, e1]
        p1: 0-2-3       [e2, e3]
        p2: 0-1-2-3     [e0, e4, e3]
        p3: 0-3         [e5]
    """

    def __init__(self):
        self.num_nodes = 4

        self.edge_index = torch.tensor(
            [
                [0, 1],
                [1, 3],
                [0, 2],
                [2, 3],
                [1, 2],
                [0, 3],
            ],
            device=DEVICE,
        )

        self.num_edges = self.edge_index.shape[0]

        # Base free-flow travel times
        # Make direct edge less dominant than before
        self.t0 = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 0.8, 1.8],
            dtype=DTYPE,
            device=DEVICE,
        )

        # Base capacities
        self.base_capacity = torch.tensor(
            [10.0, 10.0, 10.0, 10.0, 3.0, 4.0],
            dtype=DTYPE,
            device=DEVICE,
        )

        # Expandable edges: e4 and e5
        self.expandable_mask = torch.tensor(
            [0, 0, 0, 0, 1, 1],
            dtype=torch.bool,
            device=DEVICE,
        )
        self.expandable_indices = torch.where(self.expandable_mask)[0]
        self.num_expandable = int(self.expandable_mask.sum().item())

        # Paths
        self.paths = [
            [0, 1],      # p0
            [2, 3],      # p1
            [0, 4, 3],   # p2
            [5],         # p3
        ]
        self.num_paths = len(self.paths)

        # Path-edge incidence matrix A: [P, E]
        A = torch.zeros(self.num_paths, self.num_edges, dtype=DTYPE)
        for p_idx, path in enumerate(self.paths):
            for e in path:
                A[p_idx, e] = 1.0
        self.A = A.to(DEVICE)

        # Base OD demand, single OD pair 0->3
        self.base_demand = torch.tensor([12.0], dtype=DTYPE, device=DEVICE)