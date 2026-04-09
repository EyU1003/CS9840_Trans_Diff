import torch

from configs.configs import DEVICE, DTYPE
from models.model import TransportDiffusionModel


def evaluate_case(model: TransportDiffusionModel, cond_values, title: str):
    model.eval()
    with torch.no_grad():
        cond = torch.tensor([cond_values], dtype=DTYPE, device=DEVICE)
        out = model(cond)

        cap_add = out["capacity_additions"][0]
        capacities = out["capacities"][0]
        demand = out["demand"][0]
        ue_out = out["ue_outputs"][0]

        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)
        print(f"Scenario features [demand_scale, budget_signal, peak_signal]: {cond.cpu().numpy().round(3)}")
        print(f"Scaled demand:        {demand.cpu().numpy().round(3)}")
        print(f"Capacity additions:   {cap_add.cpu().numpy().round(3)}")
        print(f"Final capacities:     {capacities.cpu().numpy().round(3)}")
        print(f"Path flows:           {ue_out['path_flows'].cpu().numpy().round(3)}")
        print(f"Edge flows:           {ue_out['edge_flows'].cpu().numpy().round(3)}")
        print(f"Edge times:           {ue_out['edge_times'].cpu().numpy().round(3)}")
        print(f"Path costs:           {ue_out['path_costs'].cpu().numpy().round(3)}")
        print(f"Total travel time:    {ue_out['total_travel_time'].item():.4f}")
        print(f"Beckmann objective:   {ue_out['beckmann'].item():.4f}")


def main():
    model = TransportDiffusionModel().to(DEVICE)
    model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
    print("Loaded model from model.pt")

    evaluate_case(model, [1.0, 0.5, 0.4], "Moderate Demand Scenario")
    evaluate_case(model, [1.6, 0.8, 0.9], "High Demand Scenario")
    evaluate_case(model, [1.4, 0.1, 0.8], "Low Budget Scenario")


if __name__ == "__main__":
    main()