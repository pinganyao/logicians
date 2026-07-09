from music21 import (
    converter,
    stream,
    note,
    instrument,
    metadata,
)
from cellular_automaton import CellularAutomatonBassGenerator
import random

from music21 import stream, note, instrument


class BassMidiWriter:


    def write(self, notes, filename):

        part = stream.Part()

        part.insert(
            0,
            instrument.ElectricBass()
        )


        for n in notes:


            if n.get("rest", False):

                rest = note.Rest(
                    quarterLength=n["duration"]
                )

                part.append(rest)

                continue



            new_note = note.Note()

            new_note.pitch.midi = n["pitch"]

            new_note.duration.quarterLength = (
                n["duration"]
            )

            new_note.volume.velocity = (
                n["velocity"]
            )

            part.append(new_note)



        score = stream.Score()

        score.append(part)

        score.write(
            "midi",
            fp=filename
        )




class BassMidiReader:

    def __init__(self, filename):
        self.filename = filename

    def get_notes(self):
        score = converter.parse(self.filename)

        notes = []

        for n in score.recurse().notesAndRests:
            if n.isRest:
                notes.append(None)
            else:
                notes.append({
                    "pitch": n.pitch.midi,
                    "duration": n.duration.quarterLength,
                    "velocity": 80
                })

        return notes

import random

import random

import random


class BasslineEnhancer:


    def __init__(self, bass_notes, automaton):

        self.notes = bass_notes

        self.automaton = automaton



    def transpose(self, pitch, amount):

        return max(
            0,
            min(
                127,
                pitch + amount
            )
        )



    def add_neighbor(self, note):

        note["pitch"] = self.transpose(
            note["pitch"],
            random.choice([
                -2,
                -1,
                1,
                2
            ])
        )

        return note



    def add_fifth(self, note):

        note["pitch"] = self.transpose(
            note["pitch"],
            random.choice([
                -7,
                7
            ])
        )

        return note



    def add_octave(self, note):

        note["pitch"] = self.transpose(
            note["pitch"],
            random.choice([
                -12,
                12
            ])
        )

        return note



    def chromatic_approach(self, note):

        note["pitch"] = self.transpose(
            note["pitch"],
            random.choice([
                -1,
                1
            ])
        )

        return note



    def random_walk(self, note):

        note["pitch"] = self.transpose(
            note["pitch"],
            random.choice([
                -5,
                -4,
                -3,
                 3,
                 4,
                 5
            ])
        )

        return note



    def enhance(self):

        enhanced = []


        for i, original in enumerate(self.notes):


            if original is None:

                enhanced.append({
                    "rest": True,
                    "duration": 1
                })

                continue



            note = original.copy()


            state = self.automaton.state[
                i % self.automaton.length
            ]



            # -----------------------
            # keep
            # -----------------------

            if state == 0:

                enhanced.append(note)



            # -----------------------
            # accent
            # -----------------------

            elif state == 1:

                note["velocity"] = 120

                enhanced.append(note)



            # -----------------------
            # octave jump
            # -----------------------

            elif state == 2:

                enhanced.append(
                    self.add_octave(note)
                )



            # -----------------------
            # neighbor tone
            # -----------------------

            elif state == 3:

                enhanced.append(
                    self.add_neighbor(note)
                )



            # -----------------------
            # fifth movement
            # -----------------------

            elif state == 4:

                enhanced.append(
                    self.add_fifth(note)
                )



            # -----------------------
            # chromatic movement
            # -----------------------

            elif state == 5:

                enhanced.append(
                    self.chromatic_approach(note)
                )



            # -----------------------
            # larger melodic jump
            # -----------------------

            elif state == 6:

                enhanced.append(
                    self.random_walk(note)
                )



            # -----------------------
            # rhythmic variation
            # -----------------------

            elif state == 7:


                variation = random.choice([

                    "split",
                    "rest",
                    "repeat"

                ])



                # split note

                if variation == "split":

                    half = (
                        note["duration"]
                        /
                        2
                    )


                    first = note.copy()

                    first["duration"] = half



                    second = note.copy()

                    second["duration"] = half


                    second["pitch"] = self.transpose(
                        second["pitch"],
                        random.choice([
                            -3,
                            2,
                            4,
                            7
                        ])
                    )


                    enhanced.append(first)

                    enhanced.append(second)



                # insert rest

                elif variation == "rest":

                    half = (
                        note["duration"]
                        /
                        2
                    )


                    enhanced.append({

                        "rest": True,

                        "duration": half

                    })


                    note["duration"] = half

                    enhanced.append(note)



                # repeated note

                elif variation == "repeat":

                    half = (
                        note["duration"]
                        /
                        2
                    )


                    first = note.copy()

                    first["duration"] = half



                    second = note.copy()

                    second["duration"] = half

                    second["velocity"] = max(
                        30,
                        note["velocity"] - 20
                    )


                    enhanced.append(first)

                    enhanced.append(second)



        return enhanced

def main():

    bass_reader = BassMidiReader("bass.mid")
    bass_notes = bass_reader.get_notes()

    writer = BassMidiWriter()

    NUM_VARIATIONS = 20

    for i in range(NUM_VARIATIONS):

        automaton = CellularAutomatonBassGenerator(
            len(bass_notes)
        )

        # evolve a few generations
        for _ in range(10):
            automaton.step()

        enhancer = BasslineEnhancer(
            bass_notes,
            automaton
        )

        enhanced = enhancer.enhance()

        writer.write(
            enhanced,
            f"/Users/alfredo/Downloads/enhanced_bass_{i}.mid"
        )

        print(f"Generated variation {i}")


if __name__ == "__main__":
    main()
    print('here')
