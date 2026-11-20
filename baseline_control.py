import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2

# Load in the default scene, and create a data object to hold the current scene state
m = mujoco.MjModel.from_xml_path("scene.xml")
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

# Get mujoco's representation of the actuators
act = {m.joint(m.actuator_trnid[i, 0]).name: i for i in range(m.nu)}

# Basically the walking algorithm (right now just straight forward)
def trot(t, freq=4.5, hip_amp=0.25, knee_amp=0.35):
    ctrl = np.zeros(m.nu)

    for hip, knee, hip_dir, knee_dir, pair in LEGS.values():
        # Phase of walking motion (follows a circular pattern for walking on eact foot)
        phase = 2*np.pi*freq*t + pair*np.pi

        # Move the hip and knee joints by setting them to the next value in the circular gait
        ctrl[act[hip]] = hip_dir * hip_amp * np.sin(phase)
        ctrl[act[knee]] = knee_dir * knee_amp * max(0.0, np.cos(phase))
    return ctrl

# Mujoco viewer loop
with mujoco.viewer.launch_passive(m, d) as v:
    while v.is_running():
        t0 = time.time()
        d.ctrl[:] = trot(d.time)
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

        step += 1
        if step % 25 == 0:
            v.sync()
        time.sleep(max(0, m.opt.timestep - (time.time() - t0)))