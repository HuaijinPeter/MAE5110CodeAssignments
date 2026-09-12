import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from integrators import rk4 as integrator
from models import rimless_wheel as model


RESTING_FIXED_POINT = 0
ROLLING_LIMIT_CYCLE = 1
UNRESOLVED = 2


def simulate(initial_state, params, timestep, sim_time):
    """Simulate the full hybrid trajectory for a fixed duration."""
    num_steps = int(sim_time / timestep) + 1
    time = np.arange(num_steps) * timestep

    states = np.zeros((2, num_steps))
    states[:, 0] = initial_state

    impact_times = []
    impact_velocities = []

    for step, current_time in enumerate(time[:-1]):
        current_state = states[:, step]

        next_state = integrator.step(
            model.dynamics,
            current_time,
            current_state,
            timestep,
            params,
        )

        if model.detect_impact(next_state, params):
            next_state = model.apply_impact(
                next_state,
                params,
            )

            impact_times.append(current_time + timestep)
            impact_velocities.append(next_state[1])

        elif model.detect_backward_impact(next_state, params):
            next_state = model.apply_backward_impact(
                next_state,
                params,
            )

            impact_times.append(current_time + timestep)
            impact_velocities.append(next_state[1])

        states[:, step + 1] = next_state

    return (
        time,
        states,
        np.array(impact_times),
        np.array(impact_velocities),
    )


def simulate_forward_impacts(
    initial_state,
    params,
    timestep,
    max_time,
    target_impacts,
):
    """
    Simulate until the wheel either:
    1. completes target_impacts forward impacts, or
    2. reverses through the backward contact boundary.
    """
    state = initial_state.copy()
    impact_velocities = []

    alpha = model.calculate_alpha(params)
    backward_contact_angle = params["slope"] - alpha

    current_time = 0.0

    while (
        current_time < max_time
        and len(impact_velocities) < target_impacts
    ):
        next_state = integrator.step(
            model.dynamics,
            current_time,
            state,
            timestep,
            params,
        )

        reversed_motion = (
            next_state[0] <= backward_contact_angle
            and next_state[1] < 0
        )

        if reversed_motion:
            break

        if model.detect_impact(next_state, params):
            next_state = model.apply_impact(
                next_state,
                params,
            )
            impact_velocities.append(next_state[1])

        state = next_state
        current_time += timestep

    return np.array(impact_velocities)


def classify_attractor(
    initial_state,
    params,
    timestep,
    max_time,
    target_impacts,
):
    """Classify convergence to rolling, rest, or neither within max_time.

    Both forward and backward impacts are simulated. When the post-impact
    energy is too low to cross the upright configuration in either direction,
    the remaining plastic-impact sequence converges to the stable two-spoke
    resting configuration and is classified without numerically resolving
    infinitely many progressively smaller rocking motions.
    """
    state = initial_state.copy()
    alpha = model.calculate_alpha(params)
    forward_impact_count = 0
    current_time = 0.0

    gravity = params["gravity"]
    length = params["length"]
    collision_factor = abs(np.cos(2 * alpha))
    stable_resting_contact = abs(params["slope"]) < alpha

    def can_cross_upright(post_impact_state):
        angle = post_impact_state[0]
        angular_velocity = post_impact_state[1]
        required_velocity_squared = (
            2 * gravity / length
            * (1 - np.cos(angle))
        )
        return (
            angular_velocity**2
            > required_velocity_squared
        )

    while current_time < max_time:
        next_state = integrator.step(
            model.dynamics,
            current_time,
            state,
            timestep,
            params,
        )

        if model.detect_impact(next_state, params):
            next_state = model.apply_impact(
                next_state,
                params,
            )
            forward_impact_count += 1

            if (
                stable_resting_contact
                and not can_cross_upright(next_state)
            ):
                return RESTING_FIXED_POINT

            if forward_impact_count == target_impacts:
                return ROLLING_LIMIT_CYCLE

        elif model.detect_backward_impact(next_state, params):
            next_state = model.apply_backward_impact(
                next_state,
                params,
            )
            forward_impact_count = 0

            if (
                stable_resting_contact
                and not can_cross_upright(next_state)
            ):
                next_forward_state = np.array([
                    params["slope"] - alpha,
                    -next_state[1] * collision_factor,
                ])

                if not can_cross_upright(next_forward_state):
                    return RESTING_FIXED_POINT

        state = next_state
        current_time += timestep

    return UNRESOLVED


def calculate_attractor_fractions(attractor_map):
    """Return the sampled fraction assigned to each attractor class."""
    return {
        "resting": np.mean(
            attractor_map == RESTING_FIXED_POINT
        ),
        "rolling": np.mean(
            attractor_map == ROLLING_LIMIT_CYCLE
        ),
        "unresolved": np.mean(
            attractor_map == UNRESOLVED
        ),
    }



# Region of Attraction
def compute_roa(
    params,
    timestep,
    max_time,
    target_impacts,
    grid_size,
    max_velocity,
):
    """Estimate the rolling RoA using brute-force simulation."""
    alpha = model.calculate_alpha(params)
    slope = params["slope"]

    angle_values = np.linspace(
        slope - alpha,
        slope + alpha,
        grid_size,
    )

    velocity_values = np.linspace(
        -max_velocity,
        max_velocity,
        grid_size,
    )

    roa_map = np.full(
        (grid_size, grid_size),
        UNRESOLVED,
        dtype=int,
    )

    for velocity_index, initial_velocity in enumerate(
        velocity_values
    ):
        print(
            f"RoA row "
            f"{velocity_index + 1}/{grid_size}"
        )

        for angle_index, initial_angle in enumerate(
            angle_values
        ):
            initial_state = np.array([
                initial_angle,
                initial_velocity,
            ])

            attractor = classify_attractor(
                initial_state,
                params,
                timestep,
                max_time,
                target_impacts,
            )

            roa_map[
                velocity_index,
                angle_index,
            ] = attractor

    attractor_fractions = calculate_attractor_fractions(
        roa_map
    )

    return (
        angle_values,
        velocity_values,
        roa_map,
        attractor_fractions,
    )


def calculate_roa_boundary(angle_values, params):
    """Compute the energy-based boundary for sustained rolling."""
    gravity = params["gravity"]
    length = params["length"]
    slope = params["slope"]

    alpha = model.calculate_alpha(params)

    post_impact_angle = slope - alpha
    impact_angle = slope + alpha
    collision_factor = np.cos(2 * alpha)

    minimum_post_impact_velocity_squared = (
        2 * gravity / length
        * (1 - np.cos(post_impact_angle))
    )

    boundary_velocities = []

    for angle in angle_values:
        if angle < 0:
            upright_requirement = (
                2 * gravity / length
                * (1 - np.cos(angle))
            )
        else:
            upright_requirement = 0.0

        next_step_requirement = (
            minimum_post_impact_velocity_squared
            / collision_factor**2
            - 2 * gravity / length
            * (
                np.cos(angle)
                - np.cos(impact_angle)
            )
        )

        required_velocity_squared = max(
            0.0,
            upright_requirement,
            next_step_requirement,
        )

        boundary_velocities.append(
            np.sqrt(required_velocity_squared)
        )

    return np.array(boundary_velocities)

def evaluate_return_map(
    angular_velocity,
    params,
    timestep,
    max_time,
):
    """Map one post-impact velocity to the next post-impact velocity."""
    alpha = model.calculate_alpha(params)
    slope = params["slope"]

    initial_state = np.array([
        slope - alpha,
        angular_velocity,
    ])

    impact_velocities = simulate_forward_impacts(
        initial_state,
        params,
        timestep,
        max_time,
        target_impacts=1,
    )

    if len(impact_velocities) == 0:
        return np.nan

    return impact_velocities[0]

def compute_return_map(
    params,
    velocity_values,
    timestep,
    max_time,
):
    """Evaluate the one-step Poincare return map."""
    next_velocities = np.full(
        velocity_values.shape,
        np.nan,
    )

    for index, velocity in enumerate(velocity_values):
        next_velocities[index] = evaluate_return_map(
            velocity,
            params,
            timestep,
            max_time,
        )

    return next_velocities

def save_roa_results(
    output_dir,
    angle_values,
    velocity_values,
    roa_map,
    boundary_velocities,
    attractor_fractions,
):
    """Save RoA numerical results."""
    np.savez(
        output_dir / "roa_data.npz",
        angle_values=angle_values,
        velocity_values=velocity_values,
        roa_map=roa_map,
        boundary_velocities=boundary_velocities,
        rolling_fraction=attractor_fractions["rolling"],
        resting_fraction=attractor_fractions["resting"],
        unresolved_fraction=attractor_fractions["unresolved"],
    )

def find_return_map_fixed_point(
    velocity_values,
    next_velocities,
):
    """Find where the return map crosses the identity line."""
    valid = np.isfinite(next_velocities)

    velocities = velocity_values[valid]
    mapped_velocities = next_velocities[valid]

    difference = (
        mapped_velocities - velocities
    )

    crossing_indices = np.where(
        difference[:-1] * difference[1:] <= 0
    )[0]

    if len(crossing_indices) == 0:
        return np.nan

    index = crossing_indices[0]

    velocity_left = velocities[index]
    velocity_right = velocities[index + 1]

    difference_left = difference[index]
    difference_right = difference[index + 1]

    fixed_point = (
        velocity_left
        - difference_left
        * (velocity_right - velocity_left)
        / (difference_right - difference_left)
    )

    return fixed_point

def estimate_floquet_multiplier(
    fixed_point,
    params,
    timestep,
    max_time,
    perturbation=1e-3,
):
    velocity_left = fixed_point - perturbation
    velocity_right = fixed_point + perturbation

    mapped_left = evaluate_return_map(
        velocity_left,
        params,
        timestep,
        max_time,
    )

    mapped_right = evaluate_return_map(
        velocity_right,
        params,
        timestep,
        max_time,
    )

    floquet_multiplier = (
        mapped_right - mapped_left
    ) / (
        velocity_right - velocity_left
    )

    return floquet_multiplier

def plot_roa(
    angle_values,
    velocity_values,
    roa_map,
    boundary_velocities,
    params,
    output_dir,
):
    """Plot and save the region of attraction."""
    plt.figure()

    attractor_colors = ListedColormap([
        "tab:blue",
        "tab:orange",
        "lightgray",
    ])
    attractor_norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5],
        attractor_colors.N,
    )

    attractor_mesh = plt.pcolormesh(
        np.rad2deg(angle_values),
        velocity_values,
        roa_map,
        shading="auto",
        cmap=attractor_colors,
        norm=attractor_norm,
    )

    plt.plot(
        np.rad2deg(angle_values),
        boundary_velocities,
        color="black",
        linewidth=2,
        label="Rolling energy boundary",
    )

    alpha = model.calculate_alpha(params)
    slope = params["slope"]
    post_impact_angle = slope - alpha
    pre_impact_angle = slope + alpha

    plt.scatter(
        np.rad2deg([
            post_impact_angle,
            pre_impact_angle,
        ]),
        [0.0, 0.0],
        color="white",
        edgecolor="black",
        marker="s",
        zorder=4,
        label="Resting fixed point",
    )

    collision_factor = np.cos(2 * alpha)
    velocity_gain_squared = (
        2 * params["gravity"] / params["length"]
        * (
            np.cos(post_impact_angle)
            - np.cos(pre_impact_angle)
        )
    )
    fixed_velocity_squared = (
        collision_factor**2
        * velocity_gain_squared
        / (1 - collision_factor**2)
    )
    upright_requirement = (
        2 * params["gravity"] / params["length"]
        * (1 - np.cos(post_impact_angle))
    )

    if fixed_velocity_squared > upright_requirement:
        cycle_angles = np.linspace(
            post_impact_angle,
            pre_impact_angle,
            200,
        )
        cycle_velocities = np.sqrt(
            fixed_velocity_squared
            + 2 * params["gravity"] / params["length"]
            * (
                np.cos(post_impact_angle)
                - np.cos(cycle_angles)
            )
        )

        plt.plot(
            np.rad2deg(cycle_angles),
            cycle_velocities,
            color="white",
            linestyle="--",
            linewidth=2,
            zorder=3,
            label="Rolling limit cycle",
        )

    plt.xlabel("Initial angle (deg)")
    plt.ylabel("Initial angular velocity (rad/s)")
    plt.title(
        "Regions of Attraction"
    )

    colorbar = plt.colorbar(
        attractor_mesh,
        ticks=[0, 1, 2],
    )
    colorbar.ax.set_yticklabels([
        "Resting fixed point",
        "Rolling limit cycle",
        "Unresolved",
    ])

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "roa.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# Experiments
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

    time, states, _, _ = simulate(
        initial_state,
        params,
        timestep,
        sim_time,
    )

    plt.figure()
    plt.plot(
        time,
        np.rad2deg(states[0, :]),
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Rimless Wheel Angle")
    plt.tight_layout()

    plt.figure()
    plt.plot(
        time,
        states[1, :],
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (rad/s)")
    plt.title("Rimless Wheel Angular Velocity")
    plt.tight_layout()

    plt.show()


def run_roa_analysis():
    params = model.generate_params()

    output_dir = Path(
        "results/assignment_1/roa"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    angle_values, velocity_values, roa_map, attractor_fractions = (
        compute_roa(
            params=params,
            timestep=1e-3,
            max_time=40.0,
            target_impacts=20,
            grid_size=41,
            max_velocity=3.0,
        )
    )

    boundary_velocities = calculate_roa_boundary(
        angle_values,
        params,
    )

    print(
        f"Resting fixed-point RoA fraction = "
        f"{attractor_fractions['resting']:.3f}\n"
        f"Rolling-limit-cycle RoA fraction = "
        f"{attractor_fractions['rolling']:.3f}\n"
        f"Unresolved fraction = "
        f"{attractor_fractions['unresolved']:.3f}"
    )

    save_roa_results(
        output_dir,
        angle_values,
        velocity_values,
        roa_map,
        boundary_velocities,
        attractor_fractions,
    )

    plot_roa(
        angle_values,
        velocity_values,
        roa_map,
        boundary_velocities,
        params,
        output_dir,
    )


def run_return_map_analysis():
    params = model.generate_params()

    output_dir = Path(
        "results/assignment_1/return_map"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestep = 1e-4
    max_time = 10.0

    velocity_values = np.linspace(
        0.5,
        2.0,
        151,
    )

    next_velocities = compute_return_map(
        params,
        velocity_values,
        timestep,
        max_time,
    )

    fixed_point = find_return_map_fixed_point(
        velocity_values,
        next_velocities,
    )
    floquet_multiplier = estimate_floquet_multiplier(
        fixed_point,
        params,
        timestep =1e-5,
        max_time = 10.0,
        perturbation=1e-2,
    )

    print(
        f"Floquet multiplier = "
        f"{floquet_multiplier:.4f}"
    )
    print(
        f"Return-map fixed point = "
        f"{fixed_point:.4f} rad/s"
    )

    valid = np.isfinite(next_velocities)

    plt.figure()

    plt.plot(
        velocity_values[valid],
        next_velocities[valid],
        label="Return map",
    )

    plt.plot(
        velocity_values[valid],
        velocity_values[valid],
        linestyle="--",
        label="Identity line",
    )

    if np.isfinite(fixed_point):
        plt.scatter(
            fixed_point,
            fixed_point,
            label=(
                f"Fixed point = "
                f"{fixed_point:.3f} rad/s"
            ),
        )

    plt.xlabel(
        r"Current post-impact velocity "
        r"$\omega_n$ (rad/s)"
    )
    plt.ylabel(
        r"Next post-impact velocity "
        r"$\omega_{n+1}$ (rad/s)"
    )
    plt.title(
        "Poincaré Return Map"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "return_map.png",
        dpi=300,
        bbox_inches="tight",
    )

    np.savez(
        output_dir / "return_map_data.npz",
        velocity_values=velocity_values,
        next_velocities=next_velocities,
        fixed_point=fixed_point,
        floquet_multiplier=floquet_multiplier
    )

    plt.show()



def compute_slope_sweep(
    slope_degrees,
    timestep=1e-3,
    roa_grid_size=31,
):
    rolling_roa_fractions = []
    resting_roa_fractions = []
    unresolved_fractions = []
    fixed_points = []
    floquet_multipliers = []

    for slope_degree in slope_degrees:
        print(f"\nSlope = {slope_degree:.1f} deg")

        params = model.generate_params()
        params["slope"] = np.deg2rad(slope_degree)


        # RoA
        _, _, _, attractor_fractions = compute_roa(
            params=params,
            timestep=timestep,
            max_time=40.0,
            target_impacts=20,
            grid_size=roa_grid_size,
            max_velocity=3.0,
        )

        rolling_roa_fractions.append(
            attractor_fractions["rolling"]
        )
        resting_roa_fractions.append(
            attractor_fractions["resting"]
        )
        unresolved_fractions.append(
            attractor_fractions["unresolved"]
        )


        # Return-map fixed point
        velocity_values = np.linspace(
            0.3,
            2.5,
            121,
        )

        next_velocities = compute_return_map(
            params,
            velocity_values,
            timestep=1e-4,
            max_time=10.0,
        )

        fixed_point = find_return_map_fixed_point(
            velocity_values,
            next_velocities,
        )


        # Floquet multiplier
        fixed_points.append(fixed_point)

        if np.isfinite(fixed_point):
            floquet_multiplier = estimate_floquet_multiplier(
                fixed_point,
                params,
                timestep=1e-5,
                max_time = 10.0,
                perturbation=1e-2,
            )
        else:
            floquet_multiplier = np.nan

        floquet_multipliers.append(floquet_multiplier)

        print(
            f"rolling RoA = "
            f"{attractor_fractions['rolling']:.3f}, "
            f"resting RoA = "
            f"{attractor_fractions['resting']:.3f}, "
            f"unresolved = "
            f"{attractor_fractions['unresolved']:.3f}, "
            f"fixed point = {fixed_point:.4f}, "
            f"Floquet = {floquet_multiplier:.4f}"
        )

    return (
        np.array(rolling_roa_fractions),
        np.array(resting_roa_fractions),
        np.array(unresolved_fractions),
        np.array(fixed_points),
        np.array(floquet_multipliers),
    )

def run_slope_sweep():
    output_dir = Path(
        "results/assignment_1/slope_sweep"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    slope_degrees = np.array([
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
    ])

    (
        rolling_roa_fractions,
        resting_roa_fractions,
        unresolved_fractions,
        fixed_points,
        floquet_multipliers,
    ) = compute_slope_sweep(
        slope_degrees,
        timestep=1e-3,
        roa_grid_size=31,
    )

    params = model.generate_params()
    alpha = model.calculate_alpha(params)
    theoretical_floquet = np.cos(2 * alpha) ** 2

    # Save numerical results
    np.savez(
        output_dir / "slope_sweep_data.npz",
        slope_degrees=slope_degrees,
        roa_fractions=rolling_roa_fractions,
        rolling_roa_fractions=rolling_roa_fractions,
        resting_roa_fractions=resting_roa_fractions,
        unresolved_fractions=unresolved_fractions,
        fixed_points=fixed_points,
        floquet_multipliers=floquet_multipliers,
        theoretical_floquet=theoretical_floquet
    )

    # Plot 1: slope vs RoA
    plt.figure()

    plt.plot(
        slope_degrees,
        rolling_roa_fractions,
        marker="o",
        label="Rolling limit cycle",
    )

    plt.plot(
        slope_degrees,
        resting_roa_fractions,
        marker="o",
        label="Resting fixed point",
    )

    plt.plot(
        slope_degrees,
        unresolved_fractions,
        linestyle=":",
        marker="o",
        label="Unresolved",
    )

    plt.xlabel("Slope inclination (deg)")
    plt.ylabel("Fraction of sampled initial states")
    plt.title("Effect of Slope on Regions of Attraction")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "slope_vs_roa.png",
        dpi=300,
        bbox_inches="tight",
    )


    # Plot 2: slope vs Floquet multiplier
    plt.figure()

    plt.plot(
        slope_degrees,
        floquet_multipliers,
        marker="o",
        label="Numerical",
    )



    plt.axhline(
        theoretical_floquet,
        linestyle="--",
        label="Theory",
    )

    plt.xlabel("Slope inclination (deg)")
    plt.ylabel("Floquet multiplier")
    plt.title("Effect of Slope on Local Convergence")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "slope_vs_floquet.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


def compute_spoke_sweep(
    spoke_numbers,
    timestep=1e-3,
    roa_grid_size=31,
):
    rolling_roa_fractions = []
    resting_roa_fractions = []
    unresolved_fractions = []
    fixed_points = []
    floquet_multipliers = []
    theoretical_floquets = []

    for num_spokes in spoke_numbers:
        print(f"\nNumber of spokes = {num_spokes}")

        params = model.generate_params()
        params["num_spokes"] = int(num_spokes)

        # --------------------------------------------------
        # RoA
        # --------------------------------------------------
        _, _, _, attractor_fractions = compute_roa(
            params=params,
            timestep=timestep,
            max_time=40.0,
            target_impacts=20,
            grid_size=roa_grid_size,
            max_velocity=3.0,
        )

        rolling_roa_fractions.append(
            attractor_fractions["rolling"]
        )
        resting_roa_fractions.append(
            attractor_fractions["resting"]
        )
        unresolved_fractions.append(
            attractor_fractions["unresolved"]
        )

        # --------------------------------------------------
        # Return-map fixed point
        # --------------------------------------------------
        velocity_values = np.linspace(
            0.3,
            3.0,
            151,
        )

        next_velocities = compute_return_map(
            params,
            velocity_values,
            timestep=1e-4,
            max_time=10.0,
        )

        fixed_point = find_return_map_fixed_point(
            velocity_values,
            next_velocities,
        )

        fixed_points.append(fixed_point)

        # --------------------------------------------------
        # Numerical Floquet multiplier
        # --------------------------------------------------
        if np.isfinite(fixed_point):
            floquet_multiplier = estimate_floquet_multiplier(
                fixed_point,
                params,
                timestep=1e-5,
                max_time=10.0,
                perturbation=1e-2,
            )
        else:
            floquet_multiplier = np.nan

        floquet_multipliers.append(
            floquet_multiplier
        )

        # --------------------------------------------------
        # Theoretical Floquet multiplier
        # --------------------------------------------------
        alpha = model.calculate_alpha(params)

        theoretical_floquet = (
            np.cos(2 * alpha) ** 2
        )

        theoretical_floquets.append(
            theoretical_floquet
        )

        print(
            f"rolling RoA = "
            f"{attractor_fractions['rolling']:.3f}, "
            f"resting RoA = "
            f"{attractor_fractions['resting']:.3f}, "
            f"unresolved = "
            f"{attractor_fractions['unresolved']:.3f}, "
            f"fixed point = {fixed_point:.4f}, "
            f"Floquet = {floquet_multiplier:.4f}, "
            f"theory = {theoretical_floquet:.4f}"
        )

    return (
        np.array(rolling_roa_fractions),
        np.array(resting_roa_fractions),
        np.array(unresolved_fractions),
        np.array(fixed_points),
        np.array(floquet_multipliers),
        np.array(theoretical_floquets),
    )

def run_spoke_sweep():
    output_dir = Path(
        "results/assignment_1/spoke_sweep"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    spoke_numbers = np.arange(6, 13)

    (
        rolling_roa_fractions,
        resting_roa_fractions,
        unresolved_fractions,
        fixed_points,
        floquet_multipliers,
        theoretical_floquets,
    ) = compute_spoke_sweep(
        spoke_numbers,
        timestep=1e-3,
        roa_grid_size=31,
    )

    # Save numerical results
    np.savez(
        output_dir / "spoke_sweep_data.npz",
        spoke_numbers=spoke_numbers,
        roa_fractions=rolling_roa_fractions,
        rolling_roa_fractions=rolling_roa_fractions,
        resting_roa_fractions=resting_roa_fractions,
        unresolved_fractions=unresolved_fractions,
        fixed_points=fixed_points,
        floquet_multipliers=floquet_multipliers,
        theoretical_floquets=theoretical_floquets,
    )

    # --------------------------------------------------
    # Plot 1: number of spokes vs RoA
    # --------------------------------------------------
    plt.figure()

    plt.plot(
        spoke_numbers,
        rolling_roa_fractions,
        marker="o",
        label="Rolling limit cycle",
    )

    plt.plot(
        spoke_numbers,
        resting_roa_fractions,
        marker="o",
        label="Resting fixed point",
    )

    plt.plot(
        spoke_numbers,
        unresolved_fractions,
        linestyle=":",
        marker="o",
        label="Unresolved",
    )

    plt.xlabel("Number of spokes")
    plt.ylabel("Fraction of sampled initial states")
    plt.title(
        "Effect of Number of Spokes on Regions of Attraction"
    )

    plt.xticks(spoke_numbers)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "spokes_vs_roa.png",
        dpi=300,
        bbox_inches="tight",
    )

    # --------------------------------------------------
    # Plot 2: number of spokes vs Floquet multiplier
    # --------------------------------------------------
    plt.figure()

    valid = np.isfinite(floquet_multipliers)

    plt.plot(
        spoke_numbers[valid],
        floquet_multipliers[valid],
        marker="o",
        label="Numerical",
    )

    plt.plot(
        spoke_numbers[valid],
        theoretical_floquets[valid],
        linestyle="--",
        marker="o",
        label="Theory",
    )

    plt.xlabel("Number of spokes")
    plt.ylabel("Floquet multiplier")
    plt.title(
        "Effect of Number of Spokes on Local Convergence"
    )

    plt.xticks(spoke_numbers)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "spokes_vs_floquet.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# Command line
def main():
    parser = argparse.ArgumentParser(
        description="MAE5110 Assignment 1 experiments."
    )

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
    )

    args = parser.parse_args()

    experiments = {
        "initial": run_initial_example,
        "roa": run_roa_analysis,
        "return_map": run_return_map_analysis,
        "slope_sweep": run_slope_sweep,
        "spoke_sweep": run_spoke_sweep,
    }

    experiments[args.experiment]()


if __name__ == "__main__":
    main()
