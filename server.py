import time
import requests
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from model import get_model, INPUT_DIM
import numpy as np

# ── Federated Learning Configuration ─────────────────────────────────────────
NUM_CLIENTS  = 3
CLIENT_PORTS = [8001, 8002, 8003]
ROUNDS       = 10
LOCAL_EPOCHS = 5

global_model = get_model(input_dim=INPUT_DIM)


def load_test_data() -> DataLoader:
    df = pd.read_csv("data/server_test_data.csv")
    X = df.drop("target", axis=1).values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    dataset = TensorDataset(
        torch.tensor(X),
        torch.tensor(y).unsqueeze(1),
    )
    return DataLoader(dataset, batch_size=64, shuffle=False)


def evaluate(model: nn.Module, test_loader: DataLoader):
    model.eval()
    criterion = nn.BCELoss()
    total_loss = correct = total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            predictions = (outputs >= 0.5).float()
            correct += (predictions == targets).sum().item()
            total   += targets.size(0)
    return total_loss / total, correct / total


def run_federated_learning():
    test_loader = load_test_data()
    print("=" * 60)
    print("  Personalized Disease Risk – Federated Learning Simulation")
    print(f"  {NUM_CLIENTS} clients | {ROUNDS} rounds | {LOCAL_EPOCHS} local epochs")
    print(f"  Input features: {INPUT_DIM} clinical biomarkers")
    print("=" * 60)

    for round_num in range(1, ROUNDS + 1):
        print(f"\n── Round {round_num}/{ROUNDS} ──────────────────────────────────")

        # 1. Broadcast current global weights
        global_weights = [p.data.tolist() for p in global_model.parameters()]

        # 2. Collect locally-updated weights from each hospital client
        client_weights  = []
        client_samples  = []

        for port in CLIENT_PORTS:
            try:
                response = requests.post(
                    f"http://127.0.0.1:{port}/train",
                    json={"weights": global_weights, "epochs": LOCAL_EPOCHS, "lr": 0.01},
                    timeout=60,
                )
                if response.status_code == 200:
                    data = response.json()
                    client_weights.append(data["weights"])
                    client_samples.append(data["num_samples"])
                    print(f"  ✓ Client {data['client_id']} ({data['num_samples']:,} records) — weights received")
                else:
                    print(f"  ✗ Client on port {port}: HTTP {response.status_code}")
            except Exception as exc:
                print(f"  ✗ Client on port {port}: {exc}")

        if not client_weights:
            print("  No client responses. Aborting.")
            break

        # 3. FedAvg aggregation (weighted by dataset size)
        total_samples = sum(client_samples)
        new_global_weights = []
        for param_idx in range(len(global_weights)):
            layer_agg = np.zeros_like(np.array(global_weights[param_idx]))
            for c_idx in range(len(client_weights)):
                frac = client_samples[c_idx] / total_samples
                layer_agg += np.array(client_weights[c_idx][param_idx]) * frac
            new_global_weights.append(torch.tensor(layer_agg, dtype=torch.float32))

        # 4. Update global model
        new_state = {name: w for (name, _), w in
                     zip(global_model.named_parameters(), new_global_weights)}
        global_model.load_state_dict(new_state)

        # 5. Evaluate on held-out server test set
        loss, acc = evaluate(global_model, test_loader)
        print(f"  Global Model  →  Loss: {loss:.4f}  |  Accuracy: {acc*100:.2f}%")

    torch.save(global_model.state_dict(), "global_model.pth")
    print("\n" + "=" * 60)
    print("  Simulation complete. Global model saved → global_model.pth")
    print("=" * 60)


if __name__ == "__main__":
    run_federated_learning()
