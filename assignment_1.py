import numpy as np
import matplotlib.pyplot as plt
import argparse

from models import rimless_wheel as model
import matplotlib.pyplot as plt
from integrators import rk4 as integrator

def simulate(initial_state, params, timestep, sim_time):
    n_timesteps = int(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep

    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    impact_times = []
    impact_velocities = []

    for step, t in enumerate(time_traj[:-1]):
        current_state = state_traj[:, step]

        next_state = integrator.step(
            model.dynamics,
            t,
            current_state,
            timestep,
            params,
        )

        if model.detect_impact(next_state, params):
            next_state = model.apply_impact(
                next_state,
                params,
            )

            impact_times.append(t + timestep)
            impact_velocities.append(next_state[1])

        state_traj[:, step + 1] = next_state

    return time_traj, state_traj, impact_velocities, impact_times


def run_initial_example():
    params = model.generate_params()

    alpha = model.calculate_alpha(params)
    slope = params["slope"]

    initial_state = np.array([
        slope - alpha,
        1.2,
    ])

    timestep = 1e-3
    sim_time = 10.0

    time_traj, state_traj, impact_times, impact_velocities = simulate(
        initial_state,
        params,
        timestep,
        sim_time,
    )

    plt.figure()
    plt.plot(
        time_traj,
        np.rad2deg(state_traj[0, :]),
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Rimless Wheel Angle")
    plt.tight_layout()

    plt.figure()
    plt.plot(
        time_traj,
        state_traj[1, :],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.title("Rimless Wheel Angular Velocity")
    plt.tight_layout()

    plt.show()


# detect convergence
def converged_to_rolling(impact_velocities):
    min_impacts = 8
    window_size = 5
    tolerance = 1e-3

    if len(impact_velocities) < min_impacts:
        return False

    recent_velocities = np.array(
        impact_velocities[-window_size:]
    )

    velocity_range = (
        np.max(recent_velocities)
        - np.min(recent_velocities)
    )

    return velocity_range < tolerance


def run_roa_analysis():
    params = model.generate_params()
    alpha = model.calculate_alpha(params)
    slope = params["slope"]

    timestep = 1e-3
    sim_time = 10.0

    grid_size = 21

    angle_values = np.linspace(
        slope - alpha,
        slope + alpha,
        grid_size,
    )

    velocity_values = np.linspace(
        0.0,
        3.0,
        grid_size,
    )

    roa_map = np.zeros(
        (len(velocity_values), len(angle_values)),
        dtype=int,
    )

    failed_examples = []

    for velocity_index, initial_velocity in enumerate(
        velocity_values
    ):
        print(
            f"Row {velocity_index + 1}/{len(velocity_values)}"
        )

        for angle_index, initial_angle in enumerate(
            angle_values
        ):
            initial_state = np.array([
                initial_angle,
                initial_velocity,
            ])

            _, _, _, impact_velocities = simulate(
                initial_state,
                params,
                timestep,
                sim_time,
            )

            rolling = converged_to_rolling(
                impact_velocities
            )

            if rolling:
                roa_map[
                    velocity_index,
                    angle_index,
                ] = 1

            elif len(failed_examples) < 5:
                failed_examples.append(
                    (
                        initial_angle,
                        initial_velocity,
                    )
                )

    rolling_fraction = np.mean(roa_map)

    print(
        "Rolling RoA fraction =",
        rolling_fraction,
    )

    print("Example non-rolling initial states:")

    for angle, velocity in failed_examples:
        print(
            f"angle = {np.rad2deg(angle):.2f} deg, "
            f"velocity = {velocity:.2f} rad/s"
        )

    plt.figure()

    plt.pcolormesh(
        np.rad2deg(angle_values),
        velocity_values,
        roa_map,
        shading="auto",
    )

    plt.xlabel("Initial angle (deg)")
    plt.ylabel("Initial angular velocity (rad/s)")
    plt.title("Region of Attraction")

    colorbar = plt.colorbar()
    colorbar.set_label(
        "Converges to rolling limit cycle"
    )

    plt.tight_layout()
    plt.show()


def run_return_map_analysis():
    print("Running return-map analysis...")


def run_slope_sweep():
    print("Running slope sweep...")


def run_spoke_sweep():
    print("Running spoke sweep...")



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=[
            "initial",
            "roa",
            "return_map",
            "slope_sweep",
            "spoke_sweep",
        ],
        default="initial",
        help="Choose which experiment to run.",
    )

    args = parser.parse_args()

    if args.experiment == "initial":
        run_initial_example()

    elif args.experiment == "roa":
        run_roa_analysis()

    elif args.experiment == "return_map":
        run_return_map_analysis()

    elif args.experiment == "slope_sweep":
        run_slope_sweep()

    elif args.experiment == "spoke_sweep":
        run_spoke_sweep()


if __name__ == "__main__":
    main()
