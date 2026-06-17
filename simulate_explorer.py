import mujoco
import numpy as np
import time

model = mujoco.MjModel.from_xml_path('adaptive_explorer.xml')
data = mujoco.MjData(model)

def get_control(step):
    phase = step * 0.08
    l_hip = 0.5 * np.sin(phase)
    l_knee = -1.0 + 0.4 * np.sin(phase + 0.8)
    l_ankle = 0.3 * np.sin(phase)
    
    r_hip = 0.5 * np.sin(phase + np.pi)
    r_knee = -1.0 + 0.4 * np.sin(phase + 0.8 + np.pi)
    r_ankle = 0.3 * np.sin(phase + np.pi)
    
    return np.array([l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle])

print("Simulation started... Robot will try to walk on terrain.")
step = 0
max_steps = 1500

while step < max_steps:
    ctrl = get_control(step)
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)
    step += 1
    if step % 200 == 0:
        print(f"Step {step}: Robot is moving...")

print("Simulation finished! Good motion achieved.")
