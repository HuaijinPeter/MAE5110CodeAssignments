import timeit
import numpy as np

from models import pendulum as model
from integrators import explicit_euler
from integrators import rk4


params = {
    "gravity": 9.81,
    "length": 1,
    "mass": 0.2,
    "damping_coeff": 0.0,
}

initial_state = np.array([np.pi / 4, 0.0])


def run_simulation(integrator, timestep):
    sim_time = 5.0

    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep

    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for step, t in enumerate(time_traj[:-1]):
        state_traj[:, step + 1] = integrator.step(
            model.dynamics,
            t,
            state_traj[:, step],
            timestep,
            params,
        )

    return state_traj



# Same timestep comparison
same_dt = 2e-4

n_runs = 10

euler_same = timeit.timeit(
    lambda: run_simulation(explicit_euler, same_dt),
    number=n_runs,
) / n_runs

rk4_same = timeit.timeit(
    lambda: run_simulation(rk4, same_dt),
    number=n_runs,
) / n_runs

print("Same timestep comparison")
print(f"dt = {same_dt}")
print(f"Euler: {euler_same:.6f} s")
print(f"RK4:   {rk4_same:.6f} s")



# Compare maximum accurate timestep for each integrator
euler_max_dt = 2e-4
rk4_max_dt = 1.5e-1
euler_max = timeit.timeit(
    lambda: run_simulation(explicit_euler, euler_max_dt),
    number=n_runs,
) / n_runs

rk4_max = timeit.timeit(
    lambda: run_simulation(rk4, rk4_max_dt),
    number=n_runs,
) / n_runs

print("\nMaximum accurate timestep comparison")
print(f"Euler dt = {euler_max_dt:.6f}, time = {euler_max:.6f} s")
print(f"RK4   dt = {rk4_max_dt:.6f}, time = {rk4_max:.6f} s")
