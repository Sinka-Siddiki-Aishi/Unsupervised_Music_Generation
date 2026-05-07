import os
import random
import pretty_midi


OUTPUT_DIR = "outputs/generated_midis"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baseline_random.mid")


def generate_random_midi(output_path, num_notes=120, duration=0.25):
    """
    Random Note Generator baseline.

    This baseline does not learn from the dataset.
    It randomly selects pitches and note start times.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    current_time = 0.0

    for _ in range(num_notes):
        pitch = random.randint(48, 72)  # Middle piano range
        velocity = random.randint(60, 100)

        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=current_time,
            end=current_time + duration
        )

        instrument.notes.append(note)

        # Random rhythmic gap
        current_time += random.choice([0.125, 0.25, 0.5])

    midi.instruments.append(instrument)
    midi.write(output_path)

    print(f"Saved random baseline MIDI to {output_path}")


if __name__ == "__main__":
    generate_random_midi(OUTPUT_FILE)