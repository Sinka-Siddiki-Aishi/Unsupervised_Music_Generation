import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

from task3_transformer import Task3Transformer


TRAIN_FILE = "data/train_test_split/X_train.npy"
TEST_FILE = "data/train_test_split/X_test.npy"

OUTPUT_MODEL = "outputs/task3_transformer.pth"
PLOT_FILE = "outputs/plots/task3_transformer_loss_curve.png"
PERPLEXITY_FILE = "outputs/task3_perplexity.txt"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0005



def load_data():
    print("Loading data...")

    X_train = np.load(TRAIN_FILE, mmap_mode="r")
    X_test = np.load(TEST_FILE, mmap_mode="r")

    print("Original train shape:", X_train.shape)
    print("Original test shape:", X_test.shape)

    X_train = X_train[:]
    X_test = X_test[:]

    print("Using train shape:", X_train.shape)
    print("Using test shape:", X_test.shape)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train)
    test_dataset = TensorDataset(X_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_elements = 0

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)

            input_seq = x[:, :-1, :]
            target_seq = x[:, 1:, :]

            output = model(input_seq)

            loss = criterion(output, target_seq)

            total_loss += loss.item() * x.size(0)
            total_elements += x.size(0)

    avg_loss = total_loss / total_elements
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


def train():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = load_data()

    model = Task3Transformer(
        input_size=128,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loss_history = []
    test_loss_history = []
    perplexity_history = []

    print("Starting Task 3 Transformer training...")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)

            input_seq = x[:, :-1, :]
            target_seq = x[:, 1:, :]

            optimizer.zero_grad()

            output = model(input_seq)
            loss = criterion(output, target_seq)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        test_loss, perplexity = evaluate(
            model,
            test_loader,
            criterion,
            device
        )

        train_loss_history.append(avg_train_loss)
        test_loss_history.append(test_loss)
        perplexity_history.append(perplexity)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], "
            f"Train Loss: {avg_train_loss:.6f}, "
            f"Test Loss: {test_loss:.6f}, "
            f"Perplexity: {perplexity:.4f}"
        )

    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"Saved Transformer model to {OUTPUT_MODEL}")

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_loss_history, marker="o", label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), test_loss_history, marker="o", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Task 3 Transformer Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    plt.close()

    print(f"Saved loss curve to {PLOT_FILE}")

    final_perplexity = perplexity_history[-1]

    with open(PERPLEXITY_FILE, "w") as f:
        f.write("Task 3 Transformer Perplexity Report\n")
        f.write("------------------------------------\n")
        f.write(f"Final test loss: {test_loss_history[-1]:.6f}\n")
        f.write(f"Final perplexity: {final_perplexity:.4f}\n")

    print(f"Saved perplexity report to {PERPLEXITY_FILE}")


if __name__ == "__main__":
    train()

