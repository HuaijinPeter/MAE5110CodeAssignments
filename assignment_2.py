from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model

# Fixed controls for this visualization example.
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
}

controller_params = {
    "kp": 6.0,
    "kd": 3.0,
}


initial_state = np.array([0.0, 3.0])
initial_stance_position = np.array([0.0, 0.0])

timestep = 1e-4
sim_time = 3.0
desired_number_of_steps = 3

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# Store the world position of the current stance foot.
stance_traj = np.zeros((2, n_timesteps))
stance_traj[:, 0] = initial_stance_position

completed_steps = 0



def compute_ankle_torque(state, params, controller_params):
    """Compute a feedback-linearizing ankle torque near upright."""
    theta, angular_velocity = state

    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    kp = controller_params["kp"]
    kd = controller_params["kd"]

    # Cancel the nonlinear gravity term and impose PD dynamics:
    # theta_ddot = -kp * theta - kd * theta_dot
    gravity_cancellation = -mass * gravity * length * np.sin(theta)

    stabilizing_torque = mass * length**2 * (
        -kp * theta - kd * angular_velocity
    )

    torque = gravity_cancellation + stabilizing_torque

    torque_min = -0.1 * mass * gravity * length
    torque_max = 0.05 * mass * gravity * length

    return np.clip(torque, torque_min, torque_max)



def calculate_swing_foot_position(state, stance_position, params):
    """Return the swing-foot position in world coordinates."""
    theta = state[0]
    length = params["length"]
    angle_of_attack = params["angle_of_attack"]

    hub = stance_position + length * np.array(
        [np.sin(theta), np.cos(theta)]
    )

    swing_angle = theta - 2.0 * angle_of_attack

    swing_foot = hub - length * np.array(
        [np.sin(swing_angle), np.cos(swing_angle)]
    )

    return swing_foot




def simulate_standing_controller(
    initial_state,
    params,
    controller_params,
    timestep=1e-3,
    sim_time=5.0,
):
    """Simulate local upright stabilization with the ankle controller."""
    n_steps = round(sim_time / timestep) + 1

    time = np.arange(n_steps) * timestep
    states = np.zeros((2, n_steps))
    torques = np.zeros(n_steps)

    states[:, 0] = initial_state

    local_params = params.copy()

    for k in range(n_steps - 1):
        state = states[:, k]

        torque = compute_ankle_torque(
            state,
            local_params,
            controller_params,
        )

        local_params["ankle_torque"] = torque
        torques[k] = torque

        states[:, k + 1] = (
            state
            + timestep * model.dynamics(
                time[k],
                state,
                local_params,
            )
        )

    torques[-1] = compute_ankle_torque(
        states[:, -1],
        local_params,
        controller_params,
    )

    return time, states, torques



# Simulation loop.
for step, t in enumerate(time_traj[:-1]):
    state = state_traj[:, step]
    stance_position = stance_traj[:, step]

    # Forward Euler for now.
    next_state = state + timestep * model.dynamics(t, state, params)

    # By default, the stance foot does not move during one stance phase.
    next_stance_position = stance_position.copy()

    if model.event_guard(state, next_state, params):
        # Before relabeling the legs, find where the swing foot landed.
        next_stance_position = calculate_swing_foot_position(
            next_state,
            stance_position,
            params,
        )

        # The old swing leg becomes the new stance leg.
        next_state = model.event_dynamics(next_state, params)

        completed_steps += 1

    state_traj[:, step + 1] = next_state
    stance_traj[:, step + 1] = next_stance_position

    if completed_steps == desired_number_of_steps:
        break


# Remove unused preallocated entries.
time_traj = time_traj[: step + 2]
state_traj = state_traj[:, : step + 2]
stance_traj = stance_traj[:, : step + 2]


# Use one fixed world-coordinate camera for the whole animation.
xmin = min(-0.5, np.min(stance_traj[0]) - 0.5)
xmax = np.max(stance_traj[0]) + 1.5
ymin = np.min(stance_traj[1]) - 0.8
ymax = np.max(stance_traj[1]) + 1.5

view_limits = (xmin, xmax, ymin, ymax)


fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    model.visualize(
        state_traj[:, index],
        params,
        ax=ax,
        stance_position=stance_traj[:, index],
        view_limits=view_limits,
    )

    ax.set_title(
        f"t = {time_traj[index]:.2f} s, "
        f"x = {stance_traj[0, index]:.2f} m"
    )


# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))

if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig,
    draw_frame,
    frames=frame_indices,
    interval=1000 / fps,
    repeat=False,
)

output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)

animation.save(
    output / "walker.gif",
    writer=PillowWriter(fps=fps),
)

print(
    f"Saved {output / 'walker.gif'} "
    f"({completed_steps} footstrikes, "
    f"final x = {stance_traj[0, -1]:.2f} m)."
)


# ---------------------------------------------------------
# Local standing-controller test
# ---------------------------------------------------------

standing_initial_state = np.array([0.02, 0.0])

standing_time, standing_states, standing_torques = (
    simulate_standing_controller(
        standing_initial_state,
        params,
        controller_params,
    )
)

print("\nStanding controller test:")
print(f"Initial state: {standing_initial_state}")
print(f"Final state:   {standing_states[:, -1]}")
print(
    f"Torque range: "
    f"[{standing_torques.min():.3f}, "
    f"{standing_torques.max():.3f}] N m"
)


fig_control, ax_control = plt.subplots(
    figsize=(8, 5),
    layout="constrained",
)

ax_control.plot(
    standing_time,
    standing_states[0],
    label=r"$\theta$",
)

ax_control.plot(
    standing_time,
    standing_states[1],
    label=r"$\dot{\theta}$",
)

ax_control.axhline(0.0, color="0.5", linestyle="--")

ax_control.set(
    xlabel="Time (s)",
    ylabel="State",
    title="Standing controller stabilization",
)

ax_control.legend()

fig_control.savefig(
    output / "standing_controller.png",
    dpi=150,
)

plt.show()