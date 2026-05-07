import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Allow importing from src/models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

from task1_autoencoder import Task1LSTMAutoencoder


TRAIN_FILE = "data/train_test_split/X_train.npy"
OUTPUT_MODEL = "outputs/task1_lstm_autoencoder.pth"
PLOT_FILE = "outputs/plots/task1_loss_curve.png"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001



def load_training_data():
    print("Loading training data...")
    X_train = np.load(TRAIN_FILE, mmap_mode="r")

    print("Original train shape:", X_train.shape)

    X_train = X_train[:]

    print("Using train shape:", X_train.shape)

    X_train = torch.tensor(X_train, dtype=torch.float32)

    dataset = TensorDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return dataloader


def train():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataloader = load_training_data()

    model = Task1LSTMAutoencoder(
        input_size=128,
        hidden_size=256,
        latent_size=64,
        num_layers=1
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    print("Starting Task 1 training...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            x = batch[0].to(device)

            optimizer.zero_grad()

            reconstructed = model(x)
            loss = criterion(reconstructed, x)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"Saved model to {OUTPUT_MODEL}")

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Task 1 LSTM Autoencoder Loss Curve")
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    plt.close()

    print(f"Saved loss curve to {PLOT_FILE}")


if __name__ == "__main__":
    train()


