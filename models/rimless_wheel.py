import numpy as np

def generate_params():
    return {
        "gravity": 9.81,
        "length": 1.0,
        "mass": 0.2,
        "restitution_coeff": 0.0,
        "N": 8,
        "Incline": np.pi / 24,
    }


def generate_initial_condition():
    """Return the default post-impact state for the rimless wheel."""
    params = generate_params()
    alpha = np.pi / params["N"]
    post_impact_angle = alpha - params["Incline"]
    return np.array([-post_impact_angle, 1.0])


def dynamics(t, state, params):
    return pendulum_dynamics(t, state, params)



def pendulum_dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
        mass * gravity * length * np.sin(angle)
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative


def detect_event(previous_state, state, params):
    event_angle = np.pi / params["N"] + params["Incline"]
    return (
        previous_state[0] < event_angle
        and state[0] >= event_angle
        and state[1] > 0
    )


def reset_impact(state, params):
    alpha = np.pi / params["N"]
    post_impact_angle = alpha - params["Incline"]
    reset_state = np.array(state, copy=True)
    reset_state[0] = -post_impact_angle
    reset_state[1] = np.cos(2 * alpha) * state[1]
    return reset_state


def calculate_energy(state, height_traj, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]
    angular_velocity = state[1]

    kinetic_energy = 0.5 * mass * (length * angular_velocity) ** 2
    potential_energy = mass * gravity * (length * np.cos(angle) + height_traj)
    return kinetic_energy, potential_energy


def walking_converged(impact_velocities, window=5, tol=1e-3):
    if len(impact_velocities) < window:
        return False

    recent = np.asarray(impact_velocities[-window:])
    return np.max(np.abs(np.diff(recent))) < tol


def refine_impact(previous_state, t, coarse_dt, fine_dt, params, integrator):
    state = np.array(previous_state, copy=True)
    local_t = t

    n_fine_steps = int(np.ceil(coarse_dt / fine_dt))

    for _ in range(n_fine_steps):
        next_state = integrator(pendulum_dynamics, local_t, state, fine_dt, params)

        if detect_event(state, next_state, params):
            return next_state

        state = next_state
        local_t += fine_dt

    return state


def next_impact_velocity(
    current_velocity,
    params,
    timestep,
    impact_timestep,
    integrator,
    post_impact_angle,
    pendulum_dynamics_func,
    detect_event_func,
    refine_impact_func,
    reset_impact_func,
):
    state = np.array([-(post_impact_angle), current_velocity], dtype=float)
    t = 0.0

    while True:
        next_state = integrator(pendulum_dynamics_func, t, state, timestep, params)

        if detect_event_func(state, next_state, params):
            refined_state = refine_impact_func(
                state,
                t,
                timestep,
                impact_timestep,
                params,
                integrator,
            )
            reset_state = reset_impact_func(refined_state, params)
            return float(reset_state[1])

        state = next_state
        t += timestep

def estimate_floquet(fixed_point, epsilon, params, timestep, impact_timestep, integrator, post_impact_angle):
    P_minus = next_impact_velocity(
        fixed_point - epsilon,
        params,
        timestep,
        impact_timestep,
        integrator,
        post_impact_angle,
        pendulum_dynamics,
        detect_event,
        refine_impact,
        reset_impact,
    )
    P_plus = next_impact_velocity(
        fixed_point + epsilon,
        params,
        timestep,
        impact_timestep,
        integrator,
        post_impact_angle,
        pendulum_dynamics,
        detect_event,
        refine_impact,
        reset_impact,
    )
    return (P_plus - P_minus) / (2 * epsilon)

def compute_roa(theta_grid, omega_grid, n_timesteps, time_traj, timestep, impact_timestep, params, integrator, model):
    roa = np.zeros(theta_grid.shape, dtype=int)
    for i in range(theta_grid.shape[0]):
        for j in range(theta_grid.shape[1]):
            state_traj = np.zeros((2, n_timesteps))
            height_traj = np.zeros(n_timesteps)
            state_traj[:, 0] = [theta_grid[i, j], omega_grid[i, j]]

            impact_velocities = []
            converged = False
            failed = False

            for step, t in enumerate(time_traj[:-1]):
                current_state = state_traj[:, step]
                next_state = integrator(
                    model.pendulum_dynamics,
                    t,
                    state_traj[:, step],
                    timestep,
                    params,
                )

                state_traj[:, step + 1] = next_state

                if model.detect_event(state_traj[:, step], next_state, params):
                    impact_state = model.refine_impact(
                        state_traj[:, step],
                        t,
                        timestep,
                        impact_timestep,
                        params,
                        integrator,
                    )

                    state_traj[:, step + 1] = model.reset_impact(impact_state, params)
                    impact_velocities.append(state_traj[1, step + 1])

                    if model.walking_converged(impact_velocities):
                        converged = True
                        break
                else:
                    omega_current = current_state[1]
                    omega_next = next_state[1]

                    if omega_current >= 0 and omega_next < 0.0:
                        failed = True
                        break

            if converged:
                classification = 3
            elif failed:
                n_impacts = len(impact_velocities)
                if n_impacts == 0:
                    classification = 0
                elif n_impacts == 1:
                    classification = 1
                else:
                    classification = 2
            else:
                classification = 4

        roa[i, j] = classification

    return roa
