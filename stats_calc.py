'''
    This file's code will be used to run the model we want over all samples, and get 
    required stats such as average, min, and max distance from the ball.

    It will also be used with the deep learning model to calculate loss and accuracy. 
'''

import main_control
import numpy as np

NUM_SCENES = 25
TIMES_PER_SCENE = 4

DATA_DIR = "data"
STATS_DIR = "stats"

if __name__ == '__main__':
    # Make a stats dir if it doesn't exist
    if not Path.is_dir(STATS_DIR):
        Path(STATS_DIR).mkdir()

    # Loop through all scene data and calculate stats
    avg_dist = 0
    min_dist = 0
    max_dist = 0

    for i in range(NUM_SCENES):
        for j in range(TIMES_PER_SCENE):
            