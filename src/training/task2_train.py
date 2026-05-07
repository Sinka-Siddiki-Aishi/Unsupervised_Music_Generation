import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

from task2_vae import Task2VAE


TRAIN_FILE = "data/train_test_split/X_train.npy"
OUTPUT_MODEL = "outputs/task2_vae.pth"
PLOT_FILE = "outputs/plots/task2_vae_loss_curve.png"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
BETA = 0.001


def vae_loss_function(reconstructed, original, mu, logvar):
    reconstruction_loss = nn.functional.binary_cross_entropy(
        reconstructed,
        original,
        reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = reconstruction_loss + BETA * kl_loss

    return total_loss, reconstruction_loss, kl_loss


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

    model = Task2VAE(
        input_size=128,
        hidden_size=256,
        latent_size=64,
        num_layers=1
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_loss_history = []
    recon_loss_history = []
    kl_loss_history = []

    print("Starting Task 2 VAE training...")

    for epoch in range(EPOCHS):
        model.train()

        total_epoch_loss = 0.0
        recon_epoch_loss = 0.0
        kl_epoch_loss = 0.0

        for batch in dataloader:
            x = batch[0].to(device)

            optimizer.zero_grad()

            reconstructed, mu, logvar = model(x)

            loss, recon_loss, kl_loss = vae_loss_function(
                reconstructed,
                x,
                mu,
                logvar
            )

            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            recon_epoch_loss += recon_loss.item()
            kl_epoch_loss += kl_loss.item()

        avg_total_loss = total_epoch_loss / len(dataloader.dataset)
        avg_recon_loss = recon_epoch_loss / len(dataloader.dataset)
        avg_kl_loss = kl_epoch_loss / len(dataloader.dataset)

        total_loss_history.append(avg_total_loss)
        recon_loss_history.append(avg_recon_loss)
        kl_loss_history.append(avg_kl_loss)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}], "
            f"Total Loss: {avg_total_loss:.4f}, "
            f"Recon Loss: {avg_recon_loss:.4f}, "
            f"KL Loss: {avg_kl_loss:.4f}"
        )

    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"Saved VAE model to {OUTPUT_MODEL}")

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), total_loss_history, marker="o", label="Total Loss")
    plt.plot(range(1, EPOCHS + 1), recon_loss_history, marker="o", label="Reconstruction Loss")
    plt.plot(range(1, EPOCHS + 1), kl_loss_history, marker="o", label="KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Task 2 VAE Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    plt.close()

    print(f"Saved VAE loss curve to {PLOT_FILE}")


if __name__ == "__main__":
    train()


