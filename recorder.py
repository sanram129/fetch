'''
    This is the file to be run to generate the dataset. It allows for
    generation of a random number of scenes run by the teacher controller,
    and will save that and the ball's path, along with the camera feed, 
    joint data, and IMU data. It will also record stats for each in the teacher,
    such as the average, minimum, and maximum distances from the ball.
'''

from make_scene import make_scene
import main_control
import numpy as np
from time import time
from pathlib import Path

# How many scenes to create, as well as how many times to run a path in the same scene
NUM_SCENES = 1
TIMES_PER_SCENE = 2

SCENE_DIR = "scenes"
DATA_DIR = "data"

# Give it a scene.xml file, along with a main folder to save the data, and
# it will save time-series camera data, and return joint, and IMU data over a run using the
# teacher controller along with the ball's path
def record_scene(scene_path, data_root):
    pass

if __name__ == '__main__':
    # Create scene dir if it doesn't already exist
    if not Path.is_dir(SCENE_DIR):
        Path(SCENE_DIR).mkdir()

    # Make a data dir if it doesn't exist
    if not Path.is_dir(DATA_DIR):
        Path(DATA_DIR).mkdir()
        
    for i in range(NUM_SCENES):
        # Create a new random environment
        new_scene_path = make_scene(path=(Path(SCENE_DIR) / f'rand_scene_{i}.xml'), robot="../robot_model/Main_Robot-v8/Main_Robot-v8.xml")

        for j in range(TIMES_PER_SCENE):
            # Record the scene
            captured_data = main_control.scene_update_loop(scene=str(new_scene_path), recording_time=5)

            # Save the scene as a numpy file
            np.savez_compressed(str(Path(DATA_DIR) / f"run_scene_{i}_take_{j}.npz"),
                t=[r["t"] for r in captured_data], img=np.stack([r["img"] for r in captured_data]),
                acc=np.stack([r["acc"] for r in captured_data]), gyro=np.stack([r["gyro"] for r in captured_data]),
                q=np.stack([r["q"] for r in captured_data]), label=np.array([r["label"] for r in captured_data]),
                ball=np.stack([r["ball"] for r in captured_data]))





