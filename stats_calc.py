'''
    This file's code will be used to run the model we want over all samples, and get 
    required stats such as average and max distance from the ball.
'''

import main_control
import numpy as np
from pathlib import Path

NUM_SCENES = 25
TIMES_PER_SCENE = 4

DATA_DIR = "data"
STATS_DIR = "stats"

STATS_FILE_NAME = "teacher_stats"

if __name__ == '__main__':
    # Make a stats dir if it doesn't exist
    if not Path.is_dir(STATS_DIR):
        Path(STATS_DIR).mkdir()

    # Loop through all scene data and calculate stats
    # Opted not to calculate min since the ball is allowed to randomly get closer to the robot 
    # somewhat faster than it is able to turn; will likely be close to 0 no matter the model
    avg_dists = []
    max_dists = []

    for i in range(NUM_SCENES):
        for j in range(TIMES_PER_SCENE):
            # Load in save file
            data = np.load(str(Path(DATA_DIR) / f"run_scene_{i}_take_{j}.npz"))

            # Get distances on each frame
            dist = np.linalg.norm(data["ball"] - data["cam"], axis=1)

            # We want to get the steady-state performance after we drop the robot in and let it settle
            # so only take time points after 3 seconds
            dist = dist[data["t"] > 3]

            avg_dists.append(dist.mean())
            max_dists.append(dist.max())

    np.savez_compressed(str(Path(STATS_DIR) / f"{STATS_FILE_NAME}.npz"))
