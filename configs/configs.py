import torch

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Training
NUM_EPOCHS = 300
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5

# Scenario features
FEATURE_DIM = 3  # [demand_scale, budget_signal, peak_signal]

# Diffusion backbone
HIDDEN_DIM = 128
TIME_EMBED_DIM = 32
DIFFUSION_STEPS = 20
MAX_CAPACITY_ADDITION = 10.0

# UE layer
UE_ITERS = 40
ALPHA = 0.15
BETA = 4.0
TEMP = 6.0

# Loss
REG_WEIGHT = 0.01
BUDGET_WEIGHT = 0.2