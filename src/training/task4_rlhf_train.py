import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

from task3_transformer import Task3Transformer


TRAIN_FILE = "data/train_test_split/X_train.npy"
SURVEY_FILE = "outputs/survey_results/task4_human_scores.csv"

PRETRAINED_MODEL = "outputs/task3_transformer.pth"
OUTPUT_MODEL = "outputs/task4_rlhf_transformer.pth"
PLOT_FILE = "outputs/plots/task4_rlhf_loss_curve.png"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001
MAX_TRAIN_SAMPLES = 40000


def load_average_reward():
    """
    Reads human survey scores and converts them into a normalized reward.
    Scores are expected from 1 to 5.
    """
    scores = []

    with open(SURVEY_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            score = float(row["score"])
            scores.append(score)

    if len(scores) == 0:
        raise ValueError("No scores found in survey CSV.")

    avg_score = sum(scores) / len(scores)

    # Normalize reward to roughly 0–1
    normalized_reward = avg_score / 5.0

    print(f"Average human score: {avg_score:.4f}")
    print(f"Normalized reward: {normalized_reward:.4f}")

    return normalized_reward


def load_training_data():
    print("Loading training data...")

    X_train = np.load(TRAIN_FILE, mmap_mode="r")
    print("Original train shape:", X_train.shape)

    X_train = X_train[:]
    print("Using train shape:", X_train.shape)

    X_train = torch.tensor(X_train, dtype=torch.float32)

    dataset = TensorDataset(X_train)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return dataloader


def train_rlhf():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    reward = load_average_reward()
    dataloader = load_training_data()

    model = Task3Transformer(
        input_size=128,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)

    model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=device))
    print(f"Loaded pretrained model from {PRETRAINED_MODEL}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    print("Starting Task 4 RLHF-style fine-tuning...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            x = batch[0].to(device)

            input_seq = x[:, :-1, :]
            target_seq = x[:, 1:, :]

            optimizer.zero_grad()

            output = model(input_seq)

            base_loss = criterion(output, target_seq)

            # Reward-weighted objective:
            # Higher reward reduces loss pressure less harshly;
            # lower reward increases correction.
            rlhf_loss = base_loss * (2.0 - reward)

            rlhf_loss.backward()
            optimizer.step()

            total_loss += rlhf_loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"Epoch [{epoch + 1}/{EPOCHS}], RLHF Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), OUTPUT_MODEL)
    print(f"Saved RLHF-tuned model to {OUTPUT_MODEL}")

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), loss_history, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("RLHF Fine-Tuning Loss")
    plt.title("Task 4 RLHF Fine-Tuning Loss Curve")
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    plt.close()

    print(f"Saved RLHF loss curve to {PLOT_FILE}")


if __name__ == "__main__":
    train_rlhf()
