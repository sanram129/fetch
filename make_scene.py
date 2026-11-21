''' 
    Generates random scenes for the robot to traverse.
    Also spawns a red ball for it to track. This gets passed
    to main_control.py for the model to use later.
'''
import colorsys
import numpy as np

# Generate random colour in a range using hsv and converting to rgb
def _rgb(rng, s=(0.4, 1.0), v=(0.4, 1.0)):
    h = rng.uniform()
    r, g, b = colorsys.hsv_to_rgb(h, rng.uniform(*s), rng.uniform(*v))
    return f"{r:.3f} {g:.3f} {b:.3f}"

# Make the random scene and write it to scene_rand.xml
def make_scene(seed=None, robot="robot_model/Main_Robot-v8/Main_Robot-v8.xml", path="scene_rand.xml", far_camera=False):
    rng = np.random.default_rng(seed)

    # define the boundaries of the inside of the room
    X, Y, WH = 2.0, 1.5, 1.5

    # Also make a more matte pastel scheme for some objects
    pastel = lambda: _rgb(rng, (0.03, 1), (0.4, 0.95))
    world = []

    # Random number of lights at different positions, with different angles and diffuse values
    for _ in range(rng.integers(1, 4)):
        p = rng.uniform([-1.5, -1.0, 1.8], [1.5, 1.0, 3.0])
        dif = rng.uniform(0.25, 0.7) * rng.uniform(0.85, 1.0, 3)
        tilt = rng.uniform(-0.3, 0.3, 2)
        world.append(f'<light pos="{p[0]:.2f} {p[1]:.2f} {p[2]:.2f}" dir="{tilt[0]:.2f} {tilt[1]:.2f} -1" '
                     f'diffuse="{dif[0]:.2f} {dif[1]:.2f} {dif[2]:.2f}" '
                     f'castshadow="{rng.choice(["true", "false"])}"/>')
    amb, hdif = rng.uniform(0.1, 0.4), rng.uniform(0.2, 0.6)

    # Make the floor and give it a different friction based on rng.random
    if rng.random() < 0.6:
        tex = (f'<texture type="2d" name="tile" builtin="checker" width="300" height="300" '
               f'rgb1="{_rgb(rng, (0.05, 0.4), (0.4, 0.9))}" rgb2="{_rgb(rng, (0.05, 0.4), (0.3, 0.8))}"/>')
        mat = (f'<material name="floor" texture="tile" '
               f'texrepeat="{rng.integers(3, 12)} {rng.integers(3, 12)}" reflectance="0.05"/>')
    else:
        tex, mat = "", f'<material name="floor" rgba="{_rgb(rng, (0.05, 0.4), (0.4, 0.9))} 1"/>'
    world.append(f'<geom name="floor" type="plane" size="{X} {Y} 0.05" material="floor" '
                 f'contype="1" conaffinity="1" friction="{rng.uniform(0.6, 1.2):.2f} 0.005 0.0001"/>')

    # Make the collidable walls so we don't phase out of the scene
    for name, pos, size in [("wall_a", (-X - .05, 0, WH / 2), (.05, Y + .1, WH / 2)),
                            ("wall_b", ( X + .05, 0, WH / 2), (.05, Y + .1, WH / 2)),
                            ("wall_c", (0, -Y - .05, WH / 2), (X + .1, .05, WH / 2)),
                            ("wall_d", (0,  Y + .05, WH / 2), (X + .1, .05, WH / 2))]:
        world.append(f'<geom name="{name}" type="box" pos="{pos[0]} {pos[1]} {pos[2]}" '
                     f'size="{size[0]} {size[1]} {size[2]}" rgba="{pastel()} 1" contype="1" conaffinity="1"/>')

    # Small rugs we can walk over on the ground to change colour
    for i in range(rng.integers(1, 4)):
        hx, hy = rng.uniform(0.15, 0.5), rng.uniform(0.15, 0.4)
        world.append(f'<geom type="box" pos="{rng.uniform(-1.8, 1.8):.2f} {rng.uniform(-1.3, 1.3):.2f} {0.002 + 0.0004 * i:.4f}" '
                     f'size="{hx:.2f} {hy:.2f} 0.002" rgba="{_rgb(rng)} 1" contype="1" conaffinity="1" '
                     f'friction="{rng.uniform(0.5, 1.5):.2f} 0.005 0.0001"/>')

    # Everything else below here the robot can't collide with (baseline would have a harder time with this)
    # Small blocks
    for _ in range(rng.integers(2, 6)):
        r = rng.uniform(0.05, 0.25)
        world.append(f'<geom type="cylinder" pos="{rng.uniform(-1.8, 1.8):.2f} {rng.uniform(-1.3, 1.3):.2f} 0.0055" '
                     f'size="{r:.2f} 0.001" rgba="{_rgb(rng)} 1" contype="0" conaffinity="0"/>')

    # Additional balls
    for _ in range(rng.integers(1, 5)):
        r = rng.uniform(0.03, 0.07)
        world.append(f'<geom type="sphere" pos="{rng.uniform(-1.8, 1.8):.2f} {rng.uniform(-1.3, 1.3):.2f} {r:.3f}" '
                     f'size="{r:.3f}" rgba="{_rgb(rng)} 1" contype="0" conaffinity="0"/>')

    # Flat rectangles on walls
    for _ in range(rng.integers(2, 6)):
        hw, hh = rng.uniform(0.1, 0.4), rng.uniform(0.08, 0.25)
        z, t = rng.uniform(hh + 0.2, WH - 0.05 - hh), rng.uniform(-1, 1)
        wall = rng.integers(4)
        if wall < 2:   # walls at x = +-X
            sx = -1 if wall == 0 else 1
            pos, size = (sx * (X - 0.01), t * (Y - hw), z), (0.005, hw, hh)
        else:          # walls at y = +-Y
            sy = -1 if wall == 2 else 1
            pos, size = (t * (X - hw), sy * (Y - 0.01), z), (hw, 0.005, hh)
        world.append(f'<geom type="box" pos="{pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f}" '
                     f'size="{size[0]:.3f} {size[1]:.3f} {size[2]:.3f}" rgba="{_rgb(rng)} 1" contype="0" conaffinity="0"/>')

    # More wall furniture
    for _ in range(rng.integers(1, 4)):
        fx, fy, hh = rng.uniform(0.1, 0.25), rng.uniform(0.1, 0.25), rng.uniform(0.15, 0.45)
        t = rng.uniform(-1, 1)
        if rng.random() < 0.5:
            pos = (rng.choice([-1, 1]) * (X - fx - 0.02), t * (Y - fy), hh)
        else:
            pos = (t * (X - fx), rng.choice([-1, 1]) * (Y - fy - 0.02), hh)
        world.append(f'<geom type="box" pos="{pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f}" size="{fx:.2f} {fy:.2f} {hh:.2f}" '
                     f'rgba="{pastel()} 1" contype="0" conaffinity="0"/>')

    # Target ball
    h = rng.uniform(-0.02, 0.02) % 1.0
    br, bg, bb = colorsys.hsv_to_rgb(h, rng.uniform(0.75, 1.0), rng.uniform(0.7, 1.0))
    world.append(f'<body name="ball" mocap="true" pos="-0.9 0 0.12"><geom name="ball_geom" type="sphere" size="0.05" '
                 f'rgba="{br:.2f} {bg:.2f} {bb:.2f} 1" contype="0" conaffinity="0"/></body>')

    # Put all the objects made into the xml template, separating them by newlines
    body = "\n    ".join(world)
    xml = f"""<mujoco model="scene_rand">
  <include file="{robot}"/>
  <visual>
    <headlight ambient="{amb:.2f} {amb:.2f} {amb:.2f}" diffuse="{hdif:.2f} {hdif:.2f} {hdif:.2f}" specular="0 0 0"/>
    <global azimuth="130" elevation="-25"/>
    <map znear="0.005"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="{_rgb(rng, (0.1, 0.5), (0.5, 1.0))}" rgb2="{_rgb(rng, (0.1, 0.5), (0.05, 0.4))}" width="512" height="3072"/>
    {tex}
    {mat}
  </asset>
  <worldbody>
    {body}
  </worldbody>
</mujoco>"""

    # Write it to the scene_rand.xml file
    with open(path, "w") as f:
        f.write(xml)
    return path

if __name__ == '__main__':
    make_scene()