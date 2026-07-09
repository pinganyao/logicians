import numpy as np

class CellularAutomatonBassGenerator:
    """
    State meanings:

    0 = root
    1 = accented root
    2 = octave
    3 = neighbour tone
    4 = fifth
    5 = chromatic approach
    6 = melodic leap
    7 = rhythmic variation
    """

    def __init__(self, length: int, seed: int | None = None):
        self.length = length

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.state = np.random.randint(
            0,
            8,
            size=length,
        )

    def step(self):
        new_state = np.zeros_like(self.state)

        for i in range(self.length):
            left = self.state[(i - 1) % self.length]
            center = self.state[i]
            right = self.state[(i + 1) % self.length]

            new_state[i] = (left + center + right) % 8

        for i in range(self.length):
            if random.random() < 0.08:
                new_state[i] = random.randint(0, 7)

        self.state = new_state