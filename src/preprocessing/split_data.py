import os
import numpy as np
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/processed/piano_roll_sequences.npy"
OUTPUT_DIR = "data/train_test_split"


def split_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading processed dataset...")
    X = np.load(INPUT_FILE, mmap_mode="r")

    print("Loaded data shape:", X.shape)

    if len(X) == 0:
        raise ValueError("Dataset is empty. Run midi_parser.py first.")

    indices = np.arange(len(X))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print("Creating train set...")
    X_train = X[train_idx]

    print("Creating test set...")
    X_test = X[test_idx]

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Saved train/test files successfully.")


if __name__ == "__main__":
    split_dataset()


