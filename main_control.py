''' 
    This file sets up all the mujoco control and viewer, generates samples
    for training, and then also defines the baseline and teacher controllers.
'''

import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2

# Distance in metres we want the ball to be away from the camera ideally
# For now try setting to 15cm
BALL_OPTIM_DIST = 0.15 

# The legs were exported a bit weird from Fusion 360
# This just defines the hip joint, knee joint, and relative joint directions for each
LEGS = {
    "A": ("Revolute-28", "Revolute-31", -1, 1, 0),
    "B": ("Revolute-35", "Revolute-49", 1, -1, 1),
    "C": ("Revolute-36", "Revolute-46", 1, -1, 0),
    "D": ("Revolute-34", "Revolute-43", -1, 1, 1),
}

LEFT = {"A", "D"}

# Initialize rng
rng = np.random.default_rng()

# Set default height for the ball to be fair to the robot
BALL_Z = 0.12

# Range of positions the ball will be allowed to wander between
LO = np.array([-1.7, -1.1])
HI = np.array([-0.4,  1.1])

# Accelerometer info
G = 9.81

# Some sensor bias gathered from the real-life MPU6050 I own
# This noise will be simulated in the model
gyro_bias = np.array([-3.5, 1.4, 0.1])

# Also simulate the joints used by the 12-bit AS5600 encoder the robot will also use
JOINTS = ["Revolute-16", "Revolute-17", "Revolute-22", "Revolute-23",   # lateral hip movement 
          "Revolute-28", "Revolute-35", "Revolute-36", "Revolute-34",   # forward/backward hip movement
          "Revolute-31", "Revolute-49", "Revolute-46", "Revolute-43"]   # knees                                                

# "Read" the joints and add a bit of noise
def read_joints(m, d, rng):
    qadr = [m.joint(j).qposadr[0] for j in JOINTS]
    LSB = 2*np.pi / 4096  

    q = d.qpos[qadr] + rng.normal(0, 0.002, len(qadr))
    return np.round(q / LSB) * LSB

# Simulate reading the IMU so we have similar types of data to real life
def read_imu(d, rng):
    # MPU6050 reads acceleration in Gs (multiples of 9.81)
    acc = d.sensor("imu_acc").data / G

    # Real sensor: degrees/s, mujoco: radians/s
    gyro = np.degrees(d.sensor("imu_gyro").data)

    # Simulated jitter found experimentally, roughly 0.01g and 0.15 deg 
    # for accelerometer and gyro, respectively
    acc = acc + rng.normal(0, 0.01, 3)
    gyro = gyro + gyro_bias + rng.normal(0, 0.15, 3)

    # From the MPU datasheet, convert this to the bit representations (in LSBs)
    # that the MPU6050 actually uses
    acc = np.clip(np.round(acc  * 16384) / 16384, -2, 2)
    gyro = np.clip(np.round(gyro * 131) / 131, -250, 250)

    return acc, gyro

# The algorithm the baseline will use to walk
# The models will give it a direction and this computes joint positions
def trot(t, x, y, act, m, freq=2.0, hip_amp=0.35, knee_amp=0.45):
    """x: -1 (left) .. +1 (right), y: -1 (back) .. +1 (forward)."""
    # clamp so we stay at a max speed of 1 on diagonals
    r = np.hypot(x, y)
    if r > 1.0:
        x, y = x / r, y / r

    # If we want to turn left, make the right side move farther, and vice versa
    # If side is true we go left
    side = {True: y + x, False: y - x}
    peak = max(1.0, abs(side[True]), abs(side[False]))
    side = {k: v / peak for k, v in side.items()}

    ctrl = np.zeros(m.nu)
    for leg, (hip, knee, hip_dir, knee_dir, pair) in LEGS.items():
        scale = side[leg in LEFT]
        phase = 2*np.pi*freq*t + pair*np.pi

        # How much to lift leg in walk
        lift = min(1.0, 3*abs(scale))

        # How much to move forward/backwards
        swing = max(0.0, np.cos(phase)) 

        # Calculate new positions based on this
        ctrl[act[hip]] = hip_dir * hip_amp * scale * np.sin(phase)
        ctrl[act[knee]] = knee_dir * knee_amp * lift * swing
    return ctrl

'''
    The following two functions, find_ball and get_target, make up
    the baseline controller. trot() will be used by both. The inputs
    to both the primary model and baseline are the current frame,
    and the outputs are the forward and lateral movement velocities.
'''

# Given the frame, give a cardinal direction for where we need to go
# Forward if we need to get closer or are tracking well, backwards
# if we are too close
def find_ball(frame):
    # We first need to find the ball within the frame
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Since red hues wrap the spectrum we need to define two sections in which a valid ball could be found
    mask = cv2.inRange(frame_hsv, (0, 120, 70), (10, 255, 255)) | cv2.inRange(frame_hsv, (170, 120, 70), (180, 255, 255))

    # Find contours (blobs that could possibly be the ball)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Biggest one is what we will consider the ball
    possible_ball = max(contours, key=cv2.contourArea)

    # If it's too small just ignore it
    if cv2.contourArea(possible_ball) < 20:
        return None
    
    # Enclose in a circle and treat this as the ball, giving us the radius, and the x and y of its centre
    (cx, cy), r = cv2.minEnclosingCircle(possible_ball)
    return cx, cy, 2*r
    
def get_target(ball_pos):
    # Default return value if we need it: just keep turning right and hope we find something
    if not ball_pos:
        return 1.0, 1.0

    cx, cy, diam = ball_pos

    # we know the ball's actual radius, so using scaling, we can figure out how far away it is
    CAMERA_Y_FOV = np.radians(38)
    CAMERA_H = 480
    FOCAL_LEN = (CAMERA_H / 2) / np.tan(CAMERA_Y_FOV / 2)
    BALL_DIAMETER = 0.05 # in metres

    ball_dist = FOCAL_LEN * BALL_DIAMETER / diam

    # Figure out how far to turn based on the ball's x position
    turn_dir = (cx - 320) / 320

    # Get forward/backwards motion based on how far away we are
    fwd_dir = 0.0
    if ball_dist >= BALL_OPTIM_DIST:
        fwd_dir = 1.0
    else:
        fwd_dir = -1.0

    return turn_dir, fwd_dir

# The teacher controller that always has the ball's position
def teacher(d, ball_id, ch, cam, standoff=0.30, k_turn=1.5, k_fwd=2.0, deadband=0.04):

    # Rotation matrix of the ball relative to the chassis
    # Matmul with the camera's position relative to the ball to get the ball in the camera's coordinate frame
    R = d.xmat[ch].reshape(3, 3)
    rel = R.T @ (d.mocap_pos[ball_id] - d.cam_xpos[cam])

    # Correct direction as -x is default forwards for the robot model
    ahead, right = -rel[0], rel[1]

    # Get a distance and angle of the ball
    dist = np.hypot(ahead, right)
    rel_angle = np.arctan2(right, ahead)

    # Get turn direction based off angle and a tunable gain
    turn = np.clip(k_turn * rel_angle, -1, 1)

    # Correct for distance with proportional gain which helps in case the ball gets too close/far
    err = dist - standoff

    # If we are very close (within deadband) stop moving
    err = 0.0 if abs(err) < deadband else err - np.sign(err) * deadband

    # Turn if the ball slips behind instead of reversing, as the ball does not avoid the robot 
    # (it collides with it for the time being)
    fwd = np.clip(k_fwd * err, -1, 1) * max(0.0, np.cos(rel_angle))
    return turn, fwd

# A simple overlay to put over what the robot sees so we can understand if it finds a ball better
# This is only for visual output; nothing will train using this
def overlay(frame, res):
    # Copy the image so we can write on it and reverse colour channels 
    img = np.ascontiguousarray(frame[:, :, ::-1])
    h, w = img.shape[:2]

    # If we didn't find the ball say so in frame
    if res is None:
        cv2.putText(img, "no ball", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img

    # Use the result (from find_ball) to display a bounding circle, a crosshair at the centre, 
    # and a line halfway through the screen to tell if we are correcting properly
    cx, cy, diam = res
    p = (int(cx), int(cy))

    cv2.circle(img, p, int(diam / 2), (0, 255, 0), 2)
    cv2.drawMarker(img, p, (0, 255, 0), cv2.MARKER_CROSS, 12, 2) 
    cv2.line(img, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)  

    label = f"x={cx - w/2:+.0f} y={h/2 - cy:+.0f} d={diam:.0f}px"
    cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Modified image with overlay
    return img

# How we will move the ball around the scene 
# If random is set to false we must provide a set of coordinates and timestamps to play back
def move_ball(ball, ball_id, dt, d, speed=0.2, random=True, keyframes=None):

    if random:
        # If the ball is very close to the goal or it has been long enough,
        # update the position of the ball
        to_goal = ball["goal"] - ball["pos"]
        if d.time > ball["next"] or np.linalg.norm(to_goal) < 0.05:
            ball["goal"] = rng.uniform(LO, HI)

            # Move it after somewhere between 4 and 8 seconds
            ball["next"] = d.time + rng.uniform(4.0, 8.0)
            to_goal = ball["goal"] - ball["pos"]

        # Get the target direction vector to head towards the ball without dividing by 0 accidentally
        want = speed * to_goal / (np.linalg.norm(to_goal) + 1e-9)

        # Update velocity and position of the ball
        ball["vel"] += (want - ball["vel"]) * min(1.0, dt / 0.5)
        ball["pos"] += ball["vel"] * dt

        # Add some jitter horizontally and update position of the ball
        xy = ball["pos"] + rng.normal(0, 0.002, 2)
        d.mocap_pos[ball_id] = [xy[0], xy[1], BALL_Z]
    else:
        if keyframes is None:
            raise ValueError("playback_coords cannot be 'None' if random set to False")

        # Stop and hold if we finish the sequence
        t = min(keyframes["t"][-1], d.time)        

        # Interpolate based on current time and keyframes
        d.mocap_pos[ball_id] = [np.interp(t, keyframes["t"], keyframes["ball"][:, i]) for i in range(3)]

# What the recorder calls to create a datapoint
# Will return the ball positions in an array, along with the path in which camera frames are stored,
# and IMU and joint data in arrays too
# If it is run with playback enabled, it will simply play back a scene we give it
def scene_update_loop(scene, recording_time=30, playback=False, old_log=None):
    # Things we need to update with a new scene
    # Load in the default scene, and create a data object to hold the current scene state
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)

    # Start up two cameras, one that goes on the robot (ie. what it sees)
    # and a second to view from afar for debug
    onboard_r = mujoco.Renderer(m, height=480, width=640)
    far_r     = mujoco.Renderer(m, height=360, width=480)

    # Frame rates for the two cameras when recording data
    # 10 FPS for the model, 30 for viewing
    ONBOARD_EVERY = int(0.1 / m.opt.timestep)    
    FAR_EVERY     = int(1/30 / m.opt.timestep)  

    # Get mujoco's representation of the actuators
    act = {m.joint(m.actuator_trnid[i, 0]).name: i for i in range(m.nu)}

    # ID of the ball from scene
    ball_id = m.body("ball").mocapid[0]

    # Set an initial position and velocity for the ball, as well as what it would do next
    d.mocap_pos[ball_id] = [-0.9, 0.0, BALL_Z]
    ball = {"pos": np.array([-0.9, 0.0]), "vel": np.zeros(2),
            "goal": rng.uniform(LO, HI), "next": 0.0}

    # IDs of the main part of the chassis and camera on the robot
    ch = m.body("Chassis_Back_1").id
    cam = m.camera("onboard").id

    # Physics update step
    step = 0

    # At the start don't move until we know what's going on
    target = np.zeros(2)

    # With frames at 10hz we just need to count the num of frames before ending
    num_frames = 0

    # The logs of data which we return/save afterwards
    log = old_log if playback else []

    # Mujoco viewer loop
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            d.ctrl[:] = trot(d.time, target[0], target[1], act, m)
            move_ball(ball, ball_id, m.opt.timestep, d, random=(not playback), keyframes=old_log)
            mujoco.mj_step(m, d)

            # These two if statements basically just output the current camera view to opencv
            # for viewing in separate camera feeds or just saving for the model
            if step % ONBOARD_EVERY == 0:
                onboard_r.update_scene(d, camera="onboard")
                onboard = onboard_r.render()             
                # We can calculate the target based off this image
                ball_pos = find_ball(onboard)
                target = teacher(d, ball_id, ch, cam)#get_target(ball_pos)

                # Update the logs every camera frame to sync them up
                acc, gyro = read_imu(d, rng)
                if not playback:
                    log.append(dict(
                        t=d.time,
                        img=cv2.resize(onboard, (128, 96)),
                        acc=acc, gyro=gyro,
                        q=read_joints(m, d, rng),
                        label=target,
                        ball=d.mocap_pos[ball_id].copy(), 
                    ))

                num_frames += 1 

                cv2.imshow("onboard", overlay(onboard, ball_pos))
                cv2.waitKey(1)

            if step % FAR_EVERY == 0:
                far_r.update_scene(d, camera="far")
                cv2.imshow("far", far_r.render()[:, :, ::-1])
                cv2.waitKey(1)

            # Update frame visually every 25 steps to compute faster 
            step += 1
            if step % 25 == 0:
                v.sync()
            time.sleep(max(0, m.opt.timestep - (time.time() - t0)))

            if num_frames == recording_time * 10:
                break

    return log if not playback else []

if __name__ == '__main__':
    scene_update_loop("scenes/rand_scene_0.xml", recording_time=5, playback=True, old_log=np.load("data/run_scene_0_take_0.npz"))