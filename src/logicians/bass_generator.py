from cellular_automaton import CellularAutomatonBassGenerator


class BassGenerator:
    def __init__(self, filename):
        self.reader = BassMidiReader(filename)

    def generate(self):
        bass_notes = self.reader.get_notes()

        automaton = CellularAutomatonBassGenerator(
            len(bass_notes)
        )

        for _ in range(10):
            automaton.step()

        enhancer = BasslineEnhancer(
            bass_notes,
            automaton,
        )

        return enhancer.enhance()



def main():
    generator = BassGenerator("bass.mid")
    writer = BassMidiWriter()

    for i in range(20):
        notes = generator.generate()

        writer.write(
            notes,
            f"enhanced_bass_{i}.mid"
        )

