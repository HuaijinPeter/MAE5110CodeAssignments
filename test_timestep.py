import numpy as np
import matplotlib.pyplot as plt

from models import pendulum as model
# from integrators import rk4 as integrator
from integrators import explicit_euler as integrator


# Basic simulation of the pendulum

params = {
    "gravity": 9.81,  # gravity m/s^2)
    "length": 1,  # rod length (m)
    "mass": 0.2,  # point mass at end of rod (kg)
    "damping_coeff": 0.0,  # damping coefficient (kg*m^2/s)
}


# some set-up
initial_state = np.array([np.pi / 4, 0.0])

# timestep = 1e-5 #original


def run_simulation(timestep):
    sim_time = 5.0

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    # simulation loop
    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = integrator.step(model.dynamics, t, state_traj[:, step], timestep, params
        )

    # sanity check the energies: since there is no actuation, and no damping, total energy should stay
    # constant. If we turn on the damping coefficient, it should slowly bleed out energy until it comes to
    # a stand-still.

    kinetic_energy, potential_energy = model.calculate_energy(state_traj, params)

    total_energy = kinetic_energy + potential_energy
    relative_energy_error = (total_energy - total_energy[0]) / abs(total_energy[0])

    # set a threshold as 0.01
    stable = np.all(np.abs(relative_energy_error) < 0.01)
    print(np.max(np.abs(relative_energy_error)))
    return stable,relative_energy_error

# timesteps = [
#     1e-5,
#     5e-5,
#     1e-4,
#     2e-4, # take this for explicit euler
#     2.5e-4,
#     3e-4,
#     5e-4,

# ]

timesteps = [

    1e-1,
    1.5e-1, # take this for rk4
    2e-1,
    5e-1,


]

for timestep in timesteps:
    (
        stable,
        relative_energy_error
    ) = run_simulation(timestep)

    print(f"timestep = {timestep:.5f}, stable = {stable}")
