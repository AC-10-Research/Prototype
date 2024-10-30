import mido
import os
from mido import MidiFile, MidiTrack, Message

directory = 'arpeggios'

if not os.path.exists(directory):
    os.makedirs(directory)

def create_arpeggio_midi(notes, filename):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    track.append(Message('program_change', program=0, time=0))

    for note in notes:
        track.append(Message('note_on', note=note, velocity=64, time=0))
        track.append(Message('note_off', note=note, velocity=64, time=480))

    file_path = os.path.join(directory, filename)
    midi.save(file_path)

chords = {
    'C_major': [60, 64, 67],  # C (root), E (major third), G (perfect fifth)
    'C_minor': [60, 63, 67],  # C (root), Eb (minor third), G (perfect fifth)
    'D_major': [62, 66, 69],  # D, F#, A
    'D_minor': [62, 65, 69],  # D, F, A
    'E_major': [64, 68, 71],  # E, G#, B
    'E_minor': [64, 67, 71],  # E, G, B
    'F_major': [65, 69, 72],  # F, A, C
    'F_minor': [65, 68, 72],  # F, Ab, C
    'G_major': [67, 71, 74],  # G, B, D
    'G_minor': [67, 70, 74],  # G, Bb, D
    'A_major': [69, 73, 76],  # A, C#, E
    'A_minor': [69, 72, 76],  # A, C, E
    'B_major': [71, 75, 78],  # B, D#, F#
    'B_minor': [71, 74, 78]   # B, D, F#
}

for chords_name, notes in chords.items():
    create_arpeggio_midi(notes, f'{chords_name}.mid')
