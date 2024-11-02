import os
import random
from typing import List, Dict
from dataclasses import dataclass

from mido import MidiFile, MidiTrack, Message
from pretty_midi import PrettyMIDI

# Data class to store attributes of a note
# pitch: int between 0-127 used in MIDI to map notes to integers
# start: time at which the note starts
# duration: length of time that the note is played
# velocity: int between 0-127 used in MIDI to express how intensely a note is played
@dataclass
class NoteEvent:
    pitch: int
    start: float
    duration: float
    velocity: int

# Training Song Generator initialized with a folder of MIDI files
# Args:
#   midi_folder (str): Path to folder containing MIDI files of scales
# Raises:
#   FileNotFoundError: if midi_folder doesn't exist
class TrainingSongGenerator:
    def __init__(self, midi_folder: str):

        if not os.path.exists(midi_folder):
            raise FileNotFoundError(f"MIDI folder '{midi_folder}' not found")

        self.midi_folder = midi_folder
        self.midi_files: Dict[str, List[str]] = {}
        self.load_midi_files()

    # TSG method to load MIDI files and determine the key scale
    # Raises:
    #   IndexError: if file names aren't formatted as key_mode (e.g. C_major)
    def load_midi_files(self) -> None:
        for filename in os.listdir(self.midi_folder):
            if filename.endswith('.mid'):
                try:
                    # Extract both key and mode (major/minor) from filename
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        key = f"{parts[0]}_{parts[1]}"  #e.g. "C_minor"
                        self.midi_files.setdefault(key, []).append(
                            os.path.join(self.midi_folder, filename)
                        )
                except IndexError:
                    print(f"Warning: Skipping incorrectly formatted filename: {filename}")

    # TSG method to return the scales found after loading MIDI files
    # Returns:
    #   List[str]: list of scales found in loaded files
    def get_available_scales(self) -> List[str]:
        return list(self.midi_files.keys())

    # TSG method to read MIDI file notes and store their data as a NoteEvent
    # Args:
    #   midi_file (str): MIDI file path
    # Returns:
    #   List[NoteEvent]: List of note data found in the MIDI file as NoteEvents
    # Raises:
    #   FileNotFoundError: if midi_file doesn't exist
    @staticmethod
    def extract_notes(midi_file: str) -> List[NoteEvent]:
        if not os.path.exists(midi_file):
            raise FileNotFoundError(f"MIDI file '{midi_file}' not found")

        pm = PrettyMIDI(midi_file)
        notes = []

        for instrument in pm.instruments:
            for note in instrument.notes:
                notes.append(NoteEvent(
                    pitch=note.pitch,
                    start=note.start,
                    duration=note.end - note.start,
                    velocity=note.velocity
                ))

        return notes

    # TSG method to generate a training song based on a given scale
    # Args:
    #   key (str): the musical key for the song
    #   duration (float): the length of the song in seconds
    # Returns:
    #   MidiFile: the newly generated song
    # Raises:
    #   ValueError: if key isn't found in available MIDI files
    def generate_training_song(self, key: str, duration: float = 30) -> MidiFile:

        if key not in self.midi_files:
            raise ValueError(f"No MIDI files found for key {key}")

        tsong = MidiFile(type=1)
        track = MidiTrack()
        tsong.tracks.append(track)

        current_time = 0
        accumulated_notes = []

        while current_time < duration:
            try:
                midi_file = random.choice(self.midi_files[key])
                notes = self.extract_notes(midi_file)

                for note in notes:
                    # Apply random variations to each NoteEvent for a less monotonous song
                    varied_note = NoteEvent(
                        pitch=note.pitch,
                        start=current_time + note.start,
                        duration=note.duration * random.uniform(0.2, 1),
                        velocity=min(127, max(1, note.velocity + random.randint(-5, 5)))
                    )
                    accumulated_notes.append(varied_note)

                # Update song length
                if notes:
                    pattern_duration = max(note.start + note.duration for note in notes)
                    current_time += pattern_duration + random.uniform(0.1, 0.3)

            except Exception as e:
                print(f"Warning: Error processing pattern: {str(e)}")
                continue

        # Shuffle the array of varied notes and add them to the MIDI track
        random.shuffle(accumulated_notes)
        for note in accumulated_notes:
            track.append(Message('note_on',
                                 note=note.pitch,
                                 velocity=note.velocity,
                                 time=int(note.duration * 480)))

            track.append(Message('note_off',
                                 note=note.pitch,
                                 velocity=0,
                                 time=int(note.duration * 480)))
        return tsong

    # TSG method to generate a song for every available scale
    # Args:
    #   duration (float): Duration for each song in seconds
    #   output_dir (str): Output directory for generated songs
    def generate_all_scale_songs(self,
                                 duration: float,
                                 output_dir: str) -> None:

        os.makedirs(output_dir, exist_ok=True)

        for scale in self.get_available_scales():
            try:
                print(f"Generating song for scale: {scale}")
                tsong = self.generate_training_song(scale, duration)

                # Create filename from scale name
                filename = f"training_song_{scale}.mid"
                file_path = os.path.join(output_dir, filename)
                tsong.save(file_path)
                print(f"Successfully saved: {filename}")

            except Exception as e:
                print(f"Error generating song for scale {scale}: {str(e)}")

def main():
    try:
        # Initialize generator with scales folder
        tsg = TrainingSongGenerator("scales")

        # Generate songs for all available scales
        tsg.generate_all_scale_songs(
            duration=30,
            output_dir='training_songs'
        )
        print("All songs generated successfully!")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()