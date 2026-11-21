'''
    This is the file to be run to generate the dataset. It allows for
    generation of a random number of scenes run by the teacher controller,
    and will save that and the ball's path, along with the camera feed, 
    joint data, and IMU data. It will also record stats for each in the teacher,
    such as the average, minimum, and maximum distances from the ball.
'''

from make_scene import make_scene
import main_control

NUM_SCENES = 5

