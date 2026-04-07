import random
import numpy as np
import torch

from configs import SEED, DEVICE, NUM_EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, FEATURE_DIM
from model import TransportDiffusionModel


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_batch(batch_size: int):
    """
    scenario = [demand_scale, budget_signal, peak_signal]
    """
    demand_scale = 0.7 + 1.0 * torch.rand(batch_size, 1, device=DEVICE)
    budget_signal = torch.rand(batch_size, 1, device=DEVICE)
    peak_signal = torch.rand(batch_size, 1, device=DEVICE)
    cond = torch.cat([demand_scale, budget_signal, peak_signal], dim=-1)
    return cond


def main():
    set_seed()

    model = TransportDiffusionModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"Device: {DEVICE}")
    print("Start training...")
    print("-" * 90)

    for epoch in range(NUM_EPOCHS):
        model.train()
        cond = sample_batch(BATCH_SIZE)

        out = model(cond)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == NUM_EPOCHS - 1:
            cap_add_mean = out["capacity_additions"].mean(dim=0).detach().cpu().numpy().round(3)
            demand_mean = out["demand"].mean().item()
            print(
                f"Epoch {epoch:03d} | "
                f"loss={out['loss'].item():.4f} | "
                f"ttt={out['travel_time_loss'].item():.4f} | "
                f"budget={out['budget_loss'].item():.4f} | "
                f"reg={out['reg_loss'].item():.4f} | "
                f"mean_cap_add={cap_add_mean} | "
                f"mean_demand={demand_mean:.3f}"
            )

    torch.save(model.state_dict(), "model.pt")
    print("-" * 90)
    print("Training finished.")
    print("Saved to model.pt")


if __name__ == "__main__":
    main()