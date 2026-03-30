import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from torch.utils.data import DataLoader, TensorDataset

from model import get_model, INPUT_DIM

app = FastAPI(title="FL Client – CVD Risk Model")

# ── Client global state ───────────────────────────────────────────────────────
local_model: nn.Module = get_model(input_dim=INPUT_DIM)
client_id:   int       = 0
train_loader: DataLoader | None = None


def load_data(client_num: int) -> DataLoader:
    df = pd.read_csv(f"data/client_{client_num}_data.csv")
    X = df.drop("target", axis=1).values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    dataset = TensorDataset(
        torch.tensor(X),
        torch.tensor(y).unsqueeze(1),
    )
    return DataLoader(dataset, batch_size=32, shuffle=True)


class TrainRequest(BaseModel):
    weights: list
    epochs:  int   = 5
    lr:      float = 0.01


@app.post("/train")
async def train(request: TrainRequest):
    global local_model, train_loader, client_id

    # ── 1. Set received global weights ───────────────────────────────────────
    new_state = {}
    for (name, _), new_w in zip(local_model.named_parameters(), request.weights):
        new_state[name] = torch.tensor(new_w)
    local_model.load_state_dict(new_state)

    # ── 2. Compute pos_weight to handle class imbalance ───────────────────────
    # Count positive/negative samples in the local dataset
    all_labels = torch.cat([y for _, y in train_loader])
    n_pos = all_labels.sum().item()
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor([max(n_neg / max(n_pos, 1), 1.0)])  # weight >= 1

    # ── 3. Local training with weighted loss ──────────────────────────────────
    # BCEWithLogitsLoss = numerically stable BCE + sigmoid in one step
    # We temporarily detach the sigmoid from model outputs for this loss
    criterion = nn.BCELoss()   # model output already has Sigmoid
    optimizer = optim.Adam(local_model.parameters(), lr=request.lr)
    local_model.train()

    for _ in range(request.epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = local_model(inputs)
            # Manual weighted BCE: w_pos * y*log(p) + (1-y)*log(1-p)
            eps = 1e-7
            loss = -(
                pos_weight * targets * torch.log(outputs + eps)
                + (1 - targets) * torch.log(1 - outputs + eps)
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
            optimizer.step()

    # ── 3. Add DP noise before returning weights (privacy simulation) ─────────
    noise_multiplier = 0.005
    updated_weights = []
    for param in local_model.parameters():
        noisy = param.data + torch.randn_like(param.data) * noise_multiplier
        updated_weights.append(noisy.tolist())

    return {
        "status":      "success",
        "client_id":   client_id,
        "num_samples": len(train_loader.dataset),
        "weights":     updated_weights,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client for CVD Risk Model")
    parser.add_argument("--client_id", type=int, required=True, help="Client hospital ID")
    parser.add_argument("--port",      type=int, required=True, help="Port to listen on")
    args = parser.parse_args()

    client_id    = args.client_id
    train_loader = load_data(client_id)

    print(f"[Client {client_id}] Starting on port {args.port} "
          f"| {len(train_loader.dataset):,} patient records "
          f"| {INPUT_DIM} clinical features")

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
