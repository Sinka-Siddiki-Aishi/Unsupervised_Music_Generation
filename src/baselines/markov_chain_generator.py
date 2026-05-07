import os
import random
from collections import defaultdict
import pretty_midi


RAW_MIDI_DIR = "data/raw_midi"
OUTPUT_DIR = "outputs/generated_midis"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "baseline_markov.mid")

MAX_FILES = 100
NUM_GENERATED_NOTES = 150


def extract_notes_from_midi(midi_path):
    """
    Extract pitch sequence from a MIDI file.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = []

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue

            sorted_notes = sorted(instrument.notes, key=lambda note: note.start)

            for note in sorted_notes:
                notes.append(note.pitch)

        return notes

    except Exception:
        return []


def collect_midi_files():
    """
    Recursively collect MIDI files from the dataset folder.
    """
    midi_files = []

    for root, dirs, files in os.walk(RAW_MIDI_DIR):
        for file_name in files:
            if file_name.lower().endswith((".mid", ".midi")):
                midi_files.append(os.path.join(root, file_name))

    return midi_files[:MAX_FILES]


def build_markov_chain(midi_files):
    """
    Build first-order Markov transition table:
    current_pitch -> possible_next_pitches
    """
    transitions = defaultdict(list)

    for midi_path in midi_files:
        notes = extract_notes_from_midi(midi_path)

        if len(notes) < 2:
            continue

        for current_note, next_note in zip(notes[:-1], notes[1:]):
            transitions[current_note].append(next_note)

    return transitions


def generate_markov_sequence(transitions, num_notes=150):
    """
    Generate a pitch sequence using the Markov transition table.
    """
    if not transitions:
        raise ValueError("No Markov transitions found. Check MIDI dataset.")

    current_note = random.choice(list(transitions.keys()))
    generated = [current_note]

    for _ in range(num_notes - 1):
        possible_next_notes = transitions.get(current_note)

        if not possible_next_notes:
            current_note = random.choice(list(transitions.keys()))
        else:
            current_note = random.choice(possible_next_notes)

        generated.append(current_note)

    return generated


def save_sequence_to_midi(note_sequence, output_path):
    """
    Convert generated pitch sequence into a MIDI file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    current_time = 0.0
    duration = 0.25

    for pitch in note_sequence:
        note = pretty_midi.Note(
            velocity=90,
            pitch=int(pitch),
            start=current_time,
            end=current_time + duration
        )

        instrument.notes.append(note)
        current_time += duration

    midi.instruments.append(instrument)
    midi.write(output_path)

    print(f"Saved Markov baseline MIDI to {output_path}")


def main():
    print("Collecting MIDI files...")
    midi_files = collect_midi_files()
    print(f"Using {len(midi_files)} MIDI files for Markov baseline.")

    print("Building Markov chain...")
    transitions = build_markov_chain(midi_files)

    print(f"Number of transition states: {len(transitions)}")

    print("Generating Markov MIDI...")
    generated_sequence = generate_markov_sequence(
        transitions,
        num_notes=NUM_GENERATED_NOTES
    )

    save_sequence_to_midi(generated_sequence, OUTPUT_FILE)


if __name__ == "__main__":
    main()