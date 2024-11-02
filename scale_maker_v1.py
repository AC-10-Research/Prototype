import os
import mido
from mido import MidiFile, MidiTrack, Message

# Create a folder named 'scales_midi_files' to save the MIDI files
output_folder = 'scales'
os.makedirs(output_folder, exist_ok=True)


def create_scale_midi(notes, filename):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    track.append(Message('program_change', program=0, time=0))

    for note in notes:
        track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=note, velocity=64, time=480))

    midi.save(filename)

scales = {
    'C_major': [60, 62, 64, 65, 67, 69, 71, 72],
    'C_minor': [60, 62, 63, 65, 67, 68, 70, 72],
    'C#_major': [61, 63, 65, 66, 68, 70, 72, 73],
    'C#_minor': [61, 63, 64, 66, 68, 69, 71, 73],
    'D_major': [62, 64, 66, 67, 69, 71, 73, 74],
    'D_minor': [62, 64, 65, 67, 69, 70, 72, 74],
    'Eb_major': [63, 65, 67, 68, 70, 72, 74, 75],
    'Eb_minor': [63, 65, 66, 68, 70, 71, 73, 75],
    'E_major': [64, 66, 68, 69, 71, 73, 75, 76],
    'E_minor': [64, 66, 67, 69, 71, 72, 74, 76],
    'F_major': [66, 68, 70, 71, 73, 75, 77, 78],
    'F_minor': [65, 67, 68, 70, 72, 73, 75, 77],
    'F#_major': [66, 68, 70, 71, 73, 75, 77, 78],
    'F#_minor': [66, 68, 69, 71, 73, 74, 76, 78],
    'G_major': [67, 69, 71, 72, 74, 76, 78, 79],
    'G_minor': [67, 69, 70, 72, 74, 75, 77, 79],
    'Ab_major': [68, 70, 72, 73, 75, 77, 79, 80],
    'Ab_minor': [68, 70, 71, 73, 75, 76, 78, 80],
    'A_major': [69, 71, 73, 74, 76, 78, 80, 81],
    'A_minor': [69, 71, 72, 74, 76, 77, 79, 81],
    'Bb_major': [70, 72, 74, 75, 77, 79, 81, 82],
    'Bb_minor': [70, 72, 73, 75, 77, 78, 80, 82],
    'B_major': [71, 73, 75, 76, 78, 80, 82, 83],
    'B_minor': [71, 73, 74, 76, 78, 79, 81, 83]
}

for scale_name, notes in scales.items():
    create_scale_midi(notes, os.path.join(output_folder, f'{scale_name}.mid'))

