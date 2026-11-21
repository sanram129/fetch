import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2

import pygame
pygame.init(); pygame.joystick.init()
pad = pygame.joystick.Joystick(0)

# Load in the default scene, and create a data object to hold the current scene state
m = mujoco.MjModel.from_xml_path("classroom.xml")
d = mujoco.MjData(m)

# Start up two cameras, one that goes on the robot (ie. what it sees)
# and a second to view from afar for debug
onboard_r = mujoco.Renderer(m, height=480, width=640)
far_r     = mujoco.Renderer(m, height=360, width=480)

# Frame rates for the two cameras when recording data
# 10 FPS for the model, 30 for viewing
ONBOARD_EVERY = int(0.1 / m.opt.timestep)    
FAR_EVERY     = int(1/30 / m.opt.timestep)  

# Physical frame we are on so we know whether to update
step = 0

# The legs were exported a bit weird from Fusion 360
# This just defines the hip joint, knee joint, and relative joint directions for each
LEGS = {
    "A": ("Revolute-28", "Revolute-31", -1, 1, 0),
    "B": ("Revolute-35", "Revolute-49", 1, -1, 1),
    "C": ("Revolute-36", "Revolute-46", 1, -1, 0),
    "D": ("Revolute-34", "Revolute-43", -1, 1, 1),
}

LEFT = {"A", "D"}

# Get mujoco's representation of the actuators
act = {m.joint(m.actuator_trnid[i, 0]).name: i for i in range(m.nu)}

# The algorithm the baseline will use to walk
# OpenCV will give it a direction and this computes it
def trot(t, x, y, freq=2.0, hip_amp=0.25, knee_amp=0.35):
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

def read_stick():
    pygame.event.pump()
    return pad.get_axis(0), -pad.get_axis(1)    

stick = np.zeros(2)

# Mujoco viewer loop
RATE = 1.0
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        t0 = time.time()
        target = np.array(read_stick())
        stick += np.clip(target - stick, -RATE*m.opt.timestep, RATE*m.opt.timestep)  # smoothing
        d.ctrl[:] = trot(d.time, stick[0], stick[1])
        mujoco.mj_step(m, d)

        # These two if statements basically just output the current camera view to opencv
        # for viewing in separate camera feeds or just saving for the model
        if step % ONBOARD_EVERY == 0:
            onboard_r.update_scene(d, camera="onboard")
            onboard = onboard_r.render()             
            cv2.imshow("onboard", onboard[:, :, ::-1])
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