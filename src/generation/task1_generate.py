import os
import sys
import numpy as np
import torch
import pretty_midi


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

from task1_autoencoder import Task1LSTMAutoencoder


MODEL_PATH = "outputs/task1_lstm_autoencoder.pth"
OUTPUT_DIR = "outputs/generated_midis"

NUM_SAMPLES = 5
SEQ_LEN = 128
LATENT_SIZE = 64
THRESHOLD = 0.3


def piano_roll_to_midi(piano_roll, output_path, fs=16):
    """
    Convert piano-roll matrix to MIDI.
    Input shape: time_steps x 128
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    time_steps, num_pitches = piano_roll.shape

    for pitch in range(num_pitches):
        note_on = False
        start_time = 0

        for t in range(time_steps):
            value = piano_roll[t, pitch]

            if value > 0 and not note_on:
                note_on = True
                start_time = t / fs

            elif value == 0 and note_on:
                note_on = False
                end_time = t / fs

                if end_time > start_time:
                    note = pretty_midi.Note(
                        velocity=90,
                        pitch=pitch,
                        start=start_time,
                        end=end_time
                    )
                    instrument.notes.append(note)

        if note_on:
            end_time = time_steps / fs
            note = pretty_midi.Note(
                velocity=90,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(output_path)


def generate_samples():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Task1LSTMAutoencoder(
        input_size=128,
        hidden_size=256,
        latent_size=LATENT_SIZE,
        num_layers=1
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Generating Task 1 MIDI samples...")

    with torch.no_grad():
        for i in range(NUM_SAMPLES):
            z = torch.randn(1, LATENT_SIZE).to(device)

            generated = model.decode(z, SEQ_LEN)
            generated = generated.squeeze(0).cpu().numpy()

            binary_roll = (generated > THRESHOLD).astype(np.float32)

            output_path = os.path.join(OUTPUT_DIR, f"task1_generated_{i + 1}.mid")
            piano_roll_to_midi(binary_roll, output_path)

            print(f"Saved {output_path}")

    print("Task 1 generation complete.")


if __name__ == "__main__":
    generate_samples()


