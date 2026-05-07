import os
import numpy as np
import pretty_midi
from tqdm import tqdm


RAW_MIDI_DIR = "data/raw_midi"
PROCESSED_DIR = "data/processed"
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "piano_roll_sequences.npy")


def midi_to_piano_roll(midi_path, fs=16):
    """
    Convert a MIDI file into a binary piano-roll.
    Output shape: time_steps x 128
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        piano_roll = midi.get_piano_roll(fs=fs)

        # Convert velocity values into binary note on/off values
        piano_roll = (piano_roll > 0).astype(np.float32)

        # Change shape from 128 x time_steps to time_steps x 128
        piano_roll = piano_roll.T

        return piano_roll

    except Exception as e:
        print(f"Skipping {midi_path}: {e}")
        return None

def create_sequences(piano_roll, seq_len=128, stride=32):
    """
    Split piano-roll into fixed-length overlapping windows.
    Each sequence shape: 128 x 128.

    stride=32 creates more training samples than stride=128.
    """
    sequences = []

    for start in range(0, len(piano_roll) - seq_len, stride):
        seq = piano_roll[start:start + seq_len]

        if seq.shape == (seq_len, 128):
            sequences.append(seq)

    return sequences




def preprocess_dataset():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    all_sequences = []
    midi_files = []

    for root, dirs, files in os.walk(RAW_MIDI_DIR):
        for file_name in files:
            if file_name.lower().endswith((".mid", ".midi")):
                midi_files.append(os.path.join(root, file_name))

    print(f"Found {len(midi_files)} MIDI files.")

    if len(midi_files) == 0:
        print("No MIDI files found.")
        print("Please put .mid or .midi files inside data/raw_midi/")
        return

    for midi_path in tqdm(midi_files):
        piano_roll = midi_to_piano_roll(midi_path)

        if piano_roll is None:
            continue

        sequences = create_sequences(piano_roll)
        all_sequences.extend(sequences)

    all_sequences = np.array(all_sequences, dtype=np.float32)

    print("Final dataset shape:", all_sequences.shape)

    if len(all_sequences) == 0:
        print("No valid sequences created.")
        print("Try adding longer MIDI files.")
        return

    np.save(OUTPUT_FILE, all_sequences)

    print(f"Saved processed data to {OUTPUT_FILE}")


if __name__ == "__main__":
    preprocess_dataset()


