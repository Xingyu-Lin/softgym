import sys
import numpy as np

# Must be run from bindings/gym
sys.path.append("sim2real/")
from simtaur import SimtaurScene

def main():
    np.set_printoptions(suppress=True)
    s = SimtaurScene(randomizing=False)

    for simtaur in s.simtaurs:
        simtaur.setLegPositions(np.array(
            [1.5, 0.0 * np.pi, 1.5, 0.0 * np.pi, 1.5, 0.0 * np.pi, 1.5, 0.0 * np.pi]))
    while True:
        e = np.sin(s.frames / 10) / 2
        s.step([np.array(
            [1.5, 0.0 * np.pi, 1.5, 0.0 * np.pi, 1.5, 0.0 * np.pi, 1.5, 0.0 * np.pi])
            for _ in range(len(s.simtaurs))])

        s.env.reset()

if __name__ == "__main__":
    main()
