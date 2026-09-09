import numpy as np
import matplotlib.pyplot as plt

from models import bouncingball as model
from integrators import explicit_euler as integrator


params = {
    "gravity": 9.81,
    "restitution": 0.8,
    "mass": 0.2,
}

initial_state = np.array([1.0, 0.0])


def run_simulation(timestep):
    sim_time = 5.0

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep

    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):

        # Continuous free-flight dynamics
        next_state = integrator.step(
            model.dynamics,
            t,
            state_traj[:, step],
            timestep,
            params,
        )

        # Discrete ground impact
        next_state = model.handle_impact(
            next_state,
            params,
        )

        state_traj[:, step + 1] = next_state

    kinetic_energy, potential_energy = model.calculate_energy(
        state_traj,
        params,
    )

    total_energy = kinetic_energy + potential_energy

    return (
        time_traj,
        state_traj,
        kinetic_energy,
        potential_energy,
        total_energy,
    )

timestep = 1e-5

(
    time_traj,
    state_traj,
    kinetic_energy,
    potential_energy,
    total_energy,
) = run_simulation(timestep)




plt.figure()
plt.plot(time_traj, state_traj[0, :])
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Bouncing ball height")
plt.tight_layout()
plt.show()



plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, total_energy, label="Total energy")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")
plt.title("Bouncing ball energy")
plt.legend()
plt.tight_layout()
plt.show()
