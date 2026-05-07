import os
import sys
import numpy as np
import torch
import pretty_midi


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

from task3_transformer import Task3Transformer


MODEL_PATH = "outputs/task3_transformer.pth"
OUTPUT_DIR = "outputs/generated_midis"

NUM_SAMPLES = 10
SEQ_LEN = 256
INPUT_SIZE = 128
THRESHOLD = 0.15


def piano_roll_to_midi(piano_roll, output_path, fs=16):
    """
    Convert piano-roll matrix to MIDI.
    Input shape: time_steps x 128
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

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


def generate_with_transformer(model, device, seq_len=256):
    """
    Autoregressive-style generation.
    Starts with random seed notes and repeatedly predicts the next piano-roll step.
    """
    model.eval()

    # Start with one random time step
    current_sequence = torch.zeros(1, 1, INPUT_SIZE).to(device)

    # Add a few random seed notes in middle pitch range
    for pitch in np.random.choice(range(48, 72), size=3, replace=False):
        current_sequence[0, 0, pitch] = 1.0

    generated_steps = [current_sequence[0, 0].detach().cpu().numpy()]

    with torch.no_grad():
        for _ in range(seq_len - 1):
            output = model(current_sequence)

            next_step_probs = output[:, -1, :]
            next_step_binary = (next_step_probs > THRESHOLD).float()

            # Prevent completely silent steps
            if next_step_binary.sum() == 0:
                top_pitch = torch.argmax(next_step_probs, dim=1)
                next_step_binary[0, top_pitch] = 1.0

            generated_steps.append(next_step_binary[0].detach().cpu().numpy())

            next_step = next_step_binary.unsqueeze(1)
            current_sequence = torch.cat([current_sequence, next_step], dim=1)

    generated_roll = np.array(generated_steps, dtype=np.float32)

    return generated_roll


def generate_samples():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Task3Transformer(
        input_size=128,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Generating Task 3 Transformer MIDI samples...")

    for i in range(NUM_SAMPLES):
        generated_roll = generate_with_transformer(
            model=model,
            device=device,
            seq_len=SEQ_LEN
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"task3_transformer_generated_{i + 1}.mid"
        )

        piano_roll_to_midi(generated_roll, output_path)

        print(f"Saved {output_path}")

    print("Task 3 Transformer generation complete.")


if __name__ == "__main__":
    generate_samples()


