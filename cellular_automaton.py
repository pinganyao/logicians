import random
from enum import Enum

import numpy as np
from music21 import instrument, metadata, note, stream
import random
import numpy as np


import random
import numpy as np


import random
import numpy as np


class CellularAutomatonBassGenerator:
    """
    Generates musical variation states for a bassline.

    States:

    0 = keep original
    1 = accent
    2 = octave jump
    3 = neighbor tone
    4 = fifth movement
    5 = chromatic movement
    6 = melodic leap
    7 = rhythmic variation
    """

    def __init__(self, length):
        self.length = length

        self.state = np.random.randint(
            0,
            8,
            size=length
        )


    def step(self):

        new_state = np.zeros_like(
            self.state
        )

        for i in range(self.length):

            left = self.state[
                (i - 1) % self.length
            ]

            center = self.state[i]

            right = self.state[
                (i + 1) % self.length
            ]


            # neighbor influence
            new_state[i] = (
                left +
                center +
                right
            ) % 8


        # mutation
        for i in range(self.length):

            if random.random() < 0.08:

                new_state[i] = random.randint(
                    0,
                    7
                )


        self.state = new_state