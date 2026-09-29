import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model

# ============================================================
# Parameters
# ============================================================

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

# Admissible control bounds (Assignment 2).
ALPHA_MIN = np.pi / 8  # rad
ALPHA_MAX = np.pi / 7  # rad

TORQUE_MIN_FRACTION = -0.1  # times m * g * l
TORQUE_MAX_FRACTION = 0.05  # times m * g * l

# Look-up table resolution. `--task grid` verifies that this is the
# coarsest velocity grid whose policy still works on the true dynamics.
GRID_VELOCITY_POINTS = 49
GRID_ALPHA_POINTS = 31


# ============================================================
# Helper functions
# ============================================================

def compute_maximum_velocity(params):
    """Return the Froude-2 angular velocity that bounds the state grid."""
    return np.sqrt(2 * params["gravity"] / params["length"])


def compute_ankle_torque(state, params, controller_params):
    """Compute feedback-linearizing ankle torque near upright."""
    theta, angular_velocity = state

    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    kp = controller_params["kp"]
    kd = controller_params["kd"]

    gravity_cancellation = (
        -mass * gravity * length * np.sin(theta)
    )

    stabilizing_torque = mass * length**2 * (
        -kp * theta
        -kd * angular_velocity
    )

    torque = gravity_cancellation + stabilizing_torque

    torque_min = TORQUE_MIN_FRACTION * mass * gravity * length
    torque_max = TORQUE_MAX_FRACTION * mass * gravity * length

    return np.clip(torque, torque_min, torque_max)


def calculate_swing_foot_position(state, stance_position, params):
    """Return the swing-foot position in world coordinates."""
    theta = state[0]

    length = params["length"]
    angle_of_attack = params["angle_of_attack"]

    hub = stance_position + length * np.array([
        np.sin(theta),
        np.cos(theta),
    ])

    swing_angle = theta - 2.0 * angle_of_attack

    swing_foot = hub - length * np.array([
        np.sin(swing_angle),
        np.cos(swing_angle),
    ])

    return swing_foot


def simulate_standing_controller(
    initial_state,
    params,
    controller_params,
    timestep=1e-3,
    sim_time=5.0,
):
    """Simulate local upright stabilization."""
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
            + timestep
            * model.dynamics(
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


def is_stabilized(
    initial_state,
    params,
    controller_params,
    timestep=1e-3,
    sim_time=5.0,
    theta_tolerance=0.01,
    velocity_tolerance=0.01,
    stable_time=0.2,
):
    """Return True if the standing controller stabilizes the initial state."""

    _, states, _ = simulate_standing_controller(
        initial_state,
        params,
        controller_params,
        timestep=timestep,
        sim_time=sim_time,
    )

    # If the pendulum falls past horizontal, consider it a failure.
    if np.any(np.abs(states[0]) >= np.pi / 2):
        return False

    if not np.all(np.isfinite(states)):
        return False

    stable_steps = round(stable_time / timestep)

    final_states = states[:, -stable_steps:]

    angle_stable = np.all(
        np.abs(final_states[0]) < theta_tolerance
    )

    velocity_stable = np.all(
        np.abs(final_states[1]) < velocity_tolerance
    )

    return angle_stable and velocity_stable


def compute_roa(
    params,
    controller_params,
    num_angle_points=41,
    num_velocity_points=61,
):
    """Compute the standing controller region of attraction."""

    angle_of_attack = params["angle_of_attack"]
    incline = params["incline"]

    # Relevant stance-angle range:
    # post-impact angle -> touchdown angle
    angle_min = incline - angle_of_attack
    angle_max = incline + angle_of_attack

    # Focused local velocity range for the standing controller.
    velocity_min = -1.5
    velocity_max = 1.5

    angle_values = np.linspace(
        angle_min,
        angle_max,
        num_angle_points,
    )

    velocity_values = np.linspace(
        velocity_min,
        velocity_max,
        num_velocity_points,
    )

    roa = np.zeros(
        (num_velocity_points, num_angle_points),
        dtype=bool,
    )

    total_states = num_angle_points * num_velocity_points
    tested_states = 0

    print(
        f"Computing RoA over {total_states} initial states..."
    )

    for i, angular_velocity in enumerate(velocity_values):
        for j, theta in enumerate(angle_values):
            initial_state = np.array([
                theta,
                angular_velocity,
            ])

            roa[i, j] = is_stabilized(
                initial_state,
                params,
                controller_params,
            )

            tested_states += 1

        print(
            f"Completed velocity row "
            f"{i + 1}/{num_velocity_points}"
        )

    return angle_values, velocity_values, roa


@dataclass
class StepResult:
    """Outcome of integrating one step of the walker.

    ``crossing_velocity`` is None when the walker stalls before reaching
    the next mid-stance crossing. ``times`` and ``states`` are populated
    only when the step was asked to record its trajectory.
    """

    crossing_velocity: float | None
    end_time: float
    times: list
    states: list
    impact_index: int | None


def integrate_step_to_next_crossing(
    initial_velocity,
    angle_of_attack,
    params,
    start_time=0.0,
    timestep=1e-4,
    max_step_time=3.0,
    record_trajectory=False,
):
    """
    Integrate one step, from a theta = 0 crossing to the next one.

    The arc runs mid-stance -> touchdown -> impact -> next mid-stance.
    The walker must keep moving forwards throughout; if it stalls, the
    step has failed and ``crossing_velocity`` comes back as None.
    """

    local_params = params.copy()
    local_params["angle_of_attack"] = angle_of_attack
    local_params["ankle_torque"] = 0.0

    state = np.array([0.0, initial_velocity])

    times = []
    states = []
    impact_index = None

    def record(time, state):
        if record_trajectory:
            times.append(time)
            states.append(state.copy())

    current_time = start_time
    impact_occurred = False

    for _ in range(round(max_step_time / timestep)):
        previous_state = state

        next_state = previous_state + timestep * model.dynamics(
            current_time,
            previous_state,
            local_params,
        )

        current_time += timestep

        if next_state[1] <= 0.0:
            return StepResult(None, current_time, times, states, impact_index)

        if not impact_occurred:
            if model.event_guard(previous_state, next_state, local_params):
                next_state = model.event_dynamics(next_state, local_params)
                impact_occurred = True

                record(current_time, next_state)
                impact_index = len(states) - 1

                state = next_state
                continue

        elif previous_state[0] < 0.0 <= next_state[0]:
            crossing_velocity = interpolate_crossing_velocity(
                previous_state,
                next_state,
            )

            record(current_time, np.array([0.0, crossing_velocity]))

            return StepResult(
                crossing_velocity,
                current_time,
                times,
                states,
                impact_index,
            )

        state = next_state
        record(current_time, state)

    return StepResult(None, current_time, times, states, impact_index)


def interpolate_crossing_velocity(previous_state, next_state):
    """Interpolate the angular velocity at the theta = 0 crossing."""
    previous_theta, previous_velocity = previous_state
    next_theta, next_velocity = next_state

    fraction = -previous_theta / (next_theta - previous_theta)

    return previous_velocity + fraction * (next_velocity - previous_velocity)


def simulate_poincare_step(
    initial_velocity,
    angle_of_attack,
    params,
    timestep=1e-4,
    max_step_time=3.0,
):
    """
    Return the angular velocity at the next theta = 0 crossing.

    Returns None when the walker fails to complete the step. This is the
    return map the look-up table is built from.
    """
    return integrate_step_to_next_crossing(
        initial_velocity,
        angle_of_attack,
        params,
        timestep=timestep,
        max_step_time=max_step_time,
    ).crossing_velocity



def build_transition_table(
    params,
    num_velocity_points=GRID_VELOCITY_POINTS,
    num_alpha_points=GRID_ALPHA_POINTS,
    verbose=True,
):
    """Build the Poincare state-action transition table."""

    velocity_values = np.linspace(
        0.0,
        compute_maximum_velocity(params),
        num_velocity_points,
    )

    alpha_values = np.linspace(
        ALPHA_MIN,
        ALPHA_MAX,
        num_alpha_points,
    )

    transition_table = np.full(
        (num_alpha_points, num_velocity_points),
        np.nan,
    )

    total = num_alpha_points * num_velocity_points

    if verbose:
        print(
            f"Building transition table: "
            f"{num_velocity_points} velocity states x "
            f"{num_alpha_points} actions = {total} transitions"
        )

    for i, alpha in enumerate(alpha_values):
        for j, velocity in enumerate(velocity_values):
            next_velocity = simulate_poincare_step(
                velocity,
                alpha,
                params,
            )

            if next_velocity is not None:
                transition_table[i, j] = next_velocity

        if verbose:
            print(
                f"Completed alpha row "
                f"{i + 1}/{num_alpha_points}"
            )

    return (
        velocity_values,
        alpha_values,
        transition_table,
    )

def look_up_state_index(velocity, velocity_values):
    """
    Return the index of the grid state that upper-bounds a velocity.

    Each grid point is treated as the upper edge of its cell, so a
    look-up rounds *up* rather than to the closest point. Both the
    return map and the cost-to-go grow with speed, so the tabulated
    state is then always at least as fast as the true one and the plan
    read out of the table is a guaranteed upper bound on what the
    walker actually needs. Rounding to the closest point instead makes
    the table optimistic by up to half a cell, which is enough to plan
    a final step that lands just outside the standing controller's RoA.
    """
    index = int(np.searchsorted(velocity_values, velocity, side="left"))

    return min(index, len(velocity_values) - 1)


def look_up_nearest_state_index(velocity, velocity_values):
    """
    Return the index of the grid state closest to a velocity.

    This is the obvious rule, and it is the one that does not work here:
    see `look_up_state_index` and the grid-resolution study.
    """
    return int(np.argmin(np.abs(velocity_values - velocity)))


def is_step_feasible_over_cell(transition_table, alpha_index, state_index):
    """
    Return True when a step completes from anywhere inside a grid cell.

    Grid points are the upper edges of their cells, so the slowest state
    the cell can contain is the previous grid point. The return map is
    monotone in speed, so a step that completes from the previous grid
    point completes from everywhere in the cell. Without this check a
    plan can rely on a step that the tabulated state can just about
    make, but the slightly slower true state cannot.
    """
    if not np.isfinite(transition_table[alpha_index, state_index]):
        return False

    if state_index == 0:
        return True

    return np.isfinite(
        transition_table[alpha_index, state_index - 1]
    )


@dataclass(frozen=True)
class LookUpRule:
    """
    How a velocity that falls between grid points is resolved.

    ``index_of`` maps a velocity to a grid index.
    ``requires_cell_feasibility`` says whether a planned step must also
    complete from the slowest state the cell can hold; that check only
    makes sense when cells sit below their grid point.
    """

    name: str
    index_of: Callable
    requires_cell_feasibility: bool


CONSERVATIVE_LOOK_UP = LookUpRule(
    "round-up",
    look_up_state_index,
    requires_cell_feasibility=True,
)

NEAREST_LOOK_UP = LookUpRule(
    "nearest",
    look_up_nearest_state_index,
    requires_cell_feasibility=False,
)


def find_successor_steps(
    step_counts,
    transition_table,
    alpha_index,
    state_index,
    velocity_values,
    roa_velocity_bound,
    look_up_rule,
    must_land_below=None,
):
    """
    Return how many steps remain after taking one step, or None.

    None means the step cannot be planned with: it does not complete
    from everywhere in the cell, it lands somewhere with no known plan,
    or it violates ``must_land_below`` — which the maximum-step search
    passes to keep its forward sweep acyclic.
    """
    if look_up_rule.requires_cell_feasibility:
        if not is_step_feasible_over_cell(
            transition_table,
            alpha_index,
            state_index,
        ):
            return None

    elif not np.isfinite(transition_table[alpha_index, state_index]):
        return None

    next_velocity = transition_table[alpha_index, state_index]

    # The tabulated landing is the fastest the true one can be, so a
    # tabulated landing inside the RoA really does end the walk.
    if next_velocity <= roa_velocity_bound:
        return 0

    next_index = look_up_rule.index_of(next_velocity, velocity_values)

    if must_land_below is not None and next_index >= must_land_below:
        return None

    if step_counts[next_index] < 0:
        return None

    return step_counts[next_index]


def compute_minimum_step_policy(
    velocity_values,
    alpha_values,
    transition_table,
    roa_velocity_bound,
    look_up_rule=CONSERVATIVE_LOOK_UP,
    verbose=True,
):
    """
    Compute the minimum number of walking steps needed to reach
    the standing-controller RoA from each Poincare velocity state.
    """

    num_velocities = len(velocity_values)

    # -1 means that the state has not yet been shown reachable.
    minimum_steps = np.full(
        num_velocities,
        -1,
        dtype=int,
    )

    # Store the best alpha for each state.
    best_alpha = np.full(
        num_velocities,
        np.nan,
    )

    # --------------------------------------------------------
    # Step 0: cells that lie entirely inside the standing-controller
    # RoA. The grid point is the fastest state in its cell, so testing
    # it alone covers the whole cell.
    # --------------------------------------------------------
    minimum_steps[velocity_values <= roa_velocity_bound] = 0

    if verbose:
        print(
            f"States already in RoA: "
            f"{np.count_nonzero(minimum_steps == 0)}"
        )

    # --------------------------------------------------------
    # Backward search
    # --------------------------------------------------------
    changed = True

    while changed:
        changed = False

        for j in range(num_velocities):

            # Already solved.
            if minimum_steps[j] >= 0:
                continue

            best_candidate_steps = None
            best_candidate_alpha = None

            for i, alpha in enumerate(alpha_values):

                next_steps = find_successor_steps(
                    minimum_steps,
                    transition_table,
                    i,
                    j,
                    velocity_values,
                    roa_velocity_bound,
                    look_up_rule,
                )

                if next_steps is None:
                    continue

                candidate_steps = next_steps + 1

                if (
                    best_candidate_steps is None
                    or candidate_steps
                    < best_candidate_steps
                ):
                    best_candidate_steps = candidate_steps
                    best_candidate_alpha = alpha

            if best_candidate_steps is not None:
                minimum_steps[j] = best_candidate_steps
                best_alpha[j] = best_candidate_alpha
                changed = True

    return minimum_steps, best_alpha

def compute_maximum_step_policy(
    velocity_values,
    alpha_values,
    transition_table,
    roa_velocity_bound,
    look_up_rule=CONSERVATIVE_LOOK_UP,
):
    """
    Compute the maximum finite number of walking steps
    before reaching the standing-controller RoA.
    """

    num_velocities = len(velocity_values)

    maximum_steps = np.full(
        num_velocities,
        -1,
        dtype=int,
    )

    best_alpha = np.full(
        num_velocities,
        np.nan,
    )

    # Cells already inside the RoA require zero walking steps.
    maximum_steps[velocity_values <= roa_velocity_bound] = 0

    # Because all valid return-map transitions decrease
    # velocity in this problem, process states from low
    # velocity to high velocity.
    for j in range(num_velocities):
        if maximum_steps[j] == 0:
            continue

        best_steps = -1
        best_action = np.nan

        for i, alpha in enumerate(alpha_values):
            # The sweep runs from slow to fast, so a usable step must
            # land on an already-solved, strictly slower state.
            next_steps = find_successor_steps(
                maximum_steps,
                transition_table,
                i,
                j,
                velocity_values,
                roa_velocity_bound,
                look_up_rule,
                must_land_below=j,
            )

            if next_steps is None:
                continue

            candidate_steps = next_steps + 1

            if candidate_steps > best_steps:
                best_steps = candidate_steps
                best_action = alpha

        if best_steps >= 0:
            maximum_steps[j] = best_steps
            best_alpha[j] = best_action

    return maximum_steps, best_alpha


def simulate_policy_trajectory(
    initial_velocity,
    velocity_values,
    policy_alpha,
    params,
    controller_params,
    roa_velocity_bound,
    max_steps=30,
    standing_time=5.0,
):
    """
    Run a look-up policy in closed loop and record the full trajectory.

    The action is looked up from the velocity the simulation actually
    produces at each mid-stance crossing, not from a pre-computed list:
    the table is a feedback policy, and replaying its actions open-loop
    drifts away from the states they were chosen for. Walking stops as
    soon as the true state enters the standing controller's RoA, after
    which the ankle controller takes over.
    """

    time_history = [0.0]
    state_history = [np.array([0.0, initial_velocity])]

    poincare_indices = [0]
    impact_indices = []
    actions = []

    velocity = initial_velocity
    current_time = 0.0

    for _ in range(max_steps):
        if velocity <= roa_velocity_bound:
            break

        alpha = look_up_action(velocity, velocity_values, policy_alpha)

        if not np.isfinite(alpha):
            raise RuntimeError(
                f"The policy has no action for {velocity:.4f} rad/s."
            )

        step = integrate_step_to_next_crossing(
            velocity,
            alpha,
            params,
            start_time=current_time,
            record_trajectory=True,
        )

        if step.crossing_velocity is None:
            raise RuntimeError(
                f"The walker stalled during a step from "
                f"{velocity:.4f} rad/s at alpha = {alpha:.4f} rad."
            )

        offset = len(state_history)

        time_history.extend(step.times)
        state_history.extend(step.states)

        impact_indices.append(offset + step.impact_index)
        poincare_indices.append(len(state_history) - 1)

        actions.append(alpha)

        velocity = step.crossing_velocity
        current_time = step.end_time

    walking_end_index = len(state_history) - 1

    (
        standing_times,
        standing_states,
        _,
    ) = simulate_standing_controller(
        state_history[-1],
        params,
        controller_params,
        timestep=1e-3,
        sim_time=standing_time,
    )

    for k in range(1, standing_times.size):
        time_history.append(current_time + standing_times[k])
        state_history.append(standing_states[:, k].copy())

    return {
        "times": np.asarray(time_history),
        "states": np.asarray(state_history).T,
        "poincare_indices": np.asarray(poincare_indices),
        "impact_indices": np.asarray(impact_indices),
        "walking_end_index": walking_end_index,
        "actions": np.asarray(actions),
    }


def compute_roa_velocity_bound(
    params,
    controller_params,
    search_velocity=3.0,
    tolerance=1e-3,
):
    """Bisect for the largest theta=0 velocity the ankle controller catches."""

    def is_caught(velocity):
        return is_stabilized(
            np.array([0.0, velocity]),
            params,
            controller_params,
        )

    if not is_caught(0.0):
        raise RuntimeError(
            "The standing controller does not hold the equilibrium."
        )

    if is_caught(search_velocity):
        raise RuntimeError(
            "The search window is too small to bracket the RoA boundary."
        )

    stable_velocity = 0.0
    unstable_velocity = search_velocity

    while unstable_velocity - stable_velocity > tolerance:
        midpoint = 0.5 * (stable_velocity + unstable_velocity)

        if is_caught(midpoint):
            stable_velocity = midpoint
        else:
            unstable_velocity = midpoint

    return stable_velocity


def look_up_action(
    velocity,
    velocity_values,
    policy_alpha,
    look_up_rule=CONSERVATIVE_LOOK_UP,
):
    """Return the tabulated action for the cell that contains a velocity."""
    return policy_alpha[
        look_up_rule.index_of(velocity, velocity_values)
    ]


def count_closed_loop_steps(
    initial_velocity,
    velocity_values,
    policy_alpha,
    params,
    roa_velocity_bound,
    look_up_rule=CONSERVATIVE_LOOK_UP,
    max_steps=30,
):
    """
    Execute a gridded look-up policy on the true continuous dynamics.

    The grid is used only to look up the next action; the velocity itself
    is carried forward exactly as the simulation produces it. Returns the
    number of steps taken and whether the walker reached the RoA.
    """

    velocity = initial_velocity

    for step in range(max_steps):
        if velocity <= roa_velocity_bound:
            return step, True

        alpha = look_up_action(
            velocity,
            velocity_values,
            policy_alpha,
            look_up_rule=look_up_rule,
        )

        if not np.isfinite(alpha):
            return step, False

        next_velocity = simulate_poincare_step(
            velocity,
            alpha,
            params,
        )

        if next_velocity is None:
            return step, False

        velocity = next_velocity

    return max_steps, velocity <= roa_velocity_bound


def select_test_velocities(
    params,
    alpha_values,
    roa_velocity_bound,
    num_test_states=61,
):
    """
    Sample off-grid velocities that the walker can physically handle.

    A test velocity is kept when the walker is either already inside the
    RoA or can complete at least one step with some admissible action, so
    that a later failure can only be blamed on the grid.
    """

    velocity_max = compute_maximum_velocity(params)

    # Cell-centred samples never coincide with a linspace grid point.
    edges = np.linspace(0.0, velocity_max, num_test_states + 1)
    candidates = 0.5 * (edges[:-1] + edges[1:])

    test_velocities = []

    for velocity in candidates:
        if velocity <= roa_velocity_bound:
            test_velocities.append(velocity)
            continue

        can_step = any(
            simulate_poincare_step(velocity, alpha, params) is not None
            for alpha in alpha_values
        )

        if can_step:
            test_velocities.append(velocity)

    return np.array(test_velocities)


def score_grid_resolution(
    num_velocity_points,
    num_alpha_points,
    test_velocities,
    roa_velocity_bound,
    params,
    look_up_rule,
):
    """Build one candidate grid and measure how well its policy transfers."""

    (
        velocity_values,
        alpha_values,
        transition_table,
    ) = build_transition_table(
        params,
        num_velocity_points=num_velocity_points,
        num_alpha_points=num_alpha_points,
        verbose=False,
    )

    minimum_steps, policy_alpha = compute_minimum_step_policy(
        velocity_values,
        alpha_values,
        transition_table,
        roa_velocity_bound,
        look_up_rule=look_up_rule,
        verbose=False,
    )

    num_failures = 0
    step_errors = []

    for velocity in test_velocities:
        predicted_steps = minimum_steps[
            look_up_rule.index_of(velocity, velocity_values)
        ]

        actual_steps, reached_roa = count_closed_loop_steps(
            velocity,
            velocity_values,
            policy_alpha,
            params,
            roa_velocity_bound,
            look_up_rule=look_up_rule,
        )

        if not reached_roa or predicted_steps < 0:
            num_failures += 1
            continue

        step_errors.append(abs(actual_steps - predicted_steps))

    spacing = compute_maximum_velocity(params) / (num_velocity_points - 1)

    # None when nothing succeeded, so it is never confused with zero error.
    max_step_error = max(step_errors) if step_errors else None

    return {
        "num_velocity_points": num_velocity_points,
        "spacing": spacing,
        "num_failures": num_failures,
        "success_rate": 1.0 - num_failures / len(test_velocities),
        "max_step_error": max_step_error,
    }


def is_grid_acceptable(record):
    """
    Return True when a candidate grid meets the acceptance criterion.

    Every off-grid test state must reach the RoA under the gridded policy
    on the true dynamics, and the tabulated step count must match the
    number of steps actually taken.
    """
    return (
        record["num_failures"] == 0
        and record["max_step_error"] == 0
    )


def format_step_error(record):
    """Render a grid's step-count error, or a dash when nothing succeeded."""
    if record["max_step_error"] is None:
        return "-"

    return str(record["max_step_error"])


def evaluate_grid_resolution(
    params,
    candidate_point_counts,
    test_velocities,
    roa_velocity_bound,
    look_up_rule,
    num_alpha_points=GRID_ALPHA_POINTS,
):
    """Score every candidate velocity grid under one look-up rule."""

    print(
        f"\nLook-up rule: {look_up_rule.name}"
    )

    records = []

    for num_points in candidate_point_counts:
        record = score_grid_resolution(
            num_points,
            num_alpha_points,
            test_velocities,
            roa_velocity_bound,
            params,
            look_up_rule,
        )

        records.append(record)

        print(
            f"  {num_points:>4} points  "
            f"spacing {record['spacing']:.4f}  "
            f"failures {record['num_failures']:>3}  "
            f"max step error {format_step_error(record):>3}"
        )

    return records


def select_coarsest_acceptable_grid(records):
    """Return the fewest-point grid that meets the acceptance criterion."""
    passing = [record for record in records if is_grid_acceptable(record)]

    if not passing:
        raise RuntimeError(
            "No candidate grid satisfies the acceptance criterion."
        )

    return min(passing, key=lambda record: record["num_velocity_points"])


def print_grid_resolution_table(records):
    """Print one look-up rule's resolution sweep as a table."""
    print(
        f"\n{'points':>7} {'spacing':>9} {'failures':>9} "
        f"{'success':>9} {'step err':>9} {'verdict':>9}"
    )

    for record in records:
        verdict = "PASS" if is_grid_acceptable(record) else "FAIL"

        print(
            f"{record['num_velocity_points']:>7} "
            f"{record['spacing']:>9.4f} "
            f"{record['num_failures']:>9} "
            f"{100 * record['success_rate']:>8.1f}% "
            f"{format_step_error(record):>9} "
            f"{verdict:>9}"
        )


def plot_grid_resolution(records_by_rule, selected_points, output):
    """Plot closed-loop failures against resolution for each look-up rule."""

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    for rule_name, records in records_by_rule.items():
        ax.plot(
            [record["num_velocity_points"] for record in records],
            [record["num_failures"] for record in records],
            "o-",
            label=f"{rule_name} look-up",
        )

    ax.axvline(
        selected_points,
        linestyle="--",
        color="gray",
        label=f"Selected grid ({selected_points} points)",
    )

    ax.set(
        xlabel="Number of velocity grid points",
        ylabel="Test states that never reach the RoA",
        title="Velocity Grid Resolution Study",
    )

    ax.grid(True)
    ax.legend()

    fig.savefig(output / "grid_resolution.png", dpi=150)

    return fig


# ============================================================
# Task 8: Grid-resolution verification
# ============================================================

def run_grid_analysis():
    """Determine the coarsest velocity grid that still yields a valid policy."""

    candidate_point_counts = [
        11, 21, 31, 41, 43, 45, 47, 49, 51, 61, 81, 101,
    ]

    roa_velocity_bound = compute_roa_velocity_bound(
        params,
        controller_params,
    )

    alpha_values = np.linspace(ALPHA_MIN, ALPHA_MAX, GRID_ALPHA_POINTS)

    test_velocities = select_test_velocities(
        params,
        alpha_values,
        roa_velocity_bound,
    )

    print(
        f"RoA boundary on the Poincaré section: "
        f"{roa_velocity_bound:.4f} rad/s"
    )

    print(
        f"Testing {len(test_velocities)} off-grid velocities on "
        f"{len(candidate_point_counts)} candidate grids, under two "
        f"look-up rules."
    )

    records_by_rule = {
        rule.name: evaluate_grid_resolution(
            params,
            candidate_point_counts,
            test_velocities,
            roa_velocity_bound,
            rule,
        )
        for rule in (CONSERVATIVE_LOOK_UP, NEAREST_LOOK_UP)
    }

    print(
        "\nCriterion: every off-grid test state reaches the RoA under the "
        "gridded policy on the true dynamics, with the tabulated step "
        "count matching the steps actually taken."
    )

    for rule_name, records in records_by_rule.items():
        print(f"\nLook-up rule: {rule_name}")
        print_grid_resolution_table(records)

    coarsest = select_coarsest_acceptable_grid(
        records_by_rule[CONSERVATIVE_LOOK_UP.name]
    )

    print(
        f"\nCoarsest acceptable grid: "
        f"{coarsest['num_velocity_points']} points "
        f"(spacing {coarsest['spacing']:.4f} rad/s)"
    )

    print(
        f"Grid in use (GRID_VELOCITY_POINTS): {GRID_VELOCITY_POINTS} points"
    )

    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)

    plot_grid_resolution(
        records_by_rule,
        coarsest["num_velocity_points"],
        output,
    )

    np.savez(
        output / "grid_resolution.npz",
        point_counts=np.array(candidate_point_counts),
        roa_velocity_bound=roa_velocity_bound,
        test_velocities=test_velocities,
        selected_points=coarsest["num_velocity_points"],
        **{
            f"failures_{rule_name.replace(' ', '_')}": np.array(
                [record["num_failures"] for record in records]
            )
            for rule_name, records in records_by_rule.items()
        },
    )

    print(f"\nSaved {output / 'grid_resolution.png'}")
    print(f"Saved {output / 'grid_resolution.npz'}")

    plt.show()


    plt.show()


# ============================================================
# Task 0: Sketches
# ============================================================

def draw_walker_pose(ax, state, alpha, params, title):
    """Draw one labelled walker snapshot on a fixed world view."""
    pose_params = params.copy()
    pose_params["angle_of_attack"] = alpha

    model.visualize(
        state,
        pose_params,
        ax=ax,
        view_limits=(-1.6, 1.6, -1.2, 1.6),
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_walker_snapshots(params, output):
    """Sketch the key poses of one walking step, plus a failure mode."""

    incline = params["incline"]

    snapshots = [
        (
            np.array([0.0, 2.0]),
            ALPHA_MIN,
            ("Poincaré section: mid-stance\n" r"$\theta = 0$, $\dot\theta > 0$"),
        ),
        (
            np.array([ALPHA_MIN + incline, 1.8]),
            ALPHA_MIN,
            ("Touchdown, short step\n" r"$\theta_{TD} = \alpha_{min} + \gamma$"),
        ),
        (
            np.array([ALPHA_MAX + incline, 1.8]),
            ALPHA_MAX,
            ("Touchdown, long step\n" r"$\theta_{TD} = \alpha_{max} + \gamma$"),
        ),
        (
            np.array([incline - ALPHA_MAX + 0.05, 0.0]),
            ALPHA_MAX,
            "Failure mode: stalls before\nmid-stance and rolls back",
        ),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9, 9),
        layout="constrained",
    )

    for ax, (state, alpha, title) in zip(axes.ravel(), snapshots, strict=True):
        draw_walker_pose(ax, state, alpha, params, title)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        fontsize=9,
    )

    fig.suptitle("Walker Snapshots at Key Moments")

    fig.savefig(output / "sketch_poses.png", dpi=150)

    return fig


def plot_state_space_sketch(params, output):
    """Sketch the stance phase portrait, the guards and the impact map."""

    gravity = params["gravity"]
    length = params["length"]
    incline = params["incline"]

    theta_limit = 0.75
    velocity_limit = 4.0

    fig, ax = plt.subplots(
        figsize=(9, 6),
        layout="constrained",
    )

    # Level sets of the conserved stance energy are the stance orbits.
    theta_grid, velocity_grid = np.meshgrid(
        np.linspace(-theta_limit, theta_limit, 400),
        np.linspace(-velocity_limit, velocity_limit, 400),
    )

    energy_grid = 0.5 * velocity_grid**2 - (gravity / length) * (
        1.0 - np.cos(theta_grid)
    )

    ax.contour(
        theta_grid,
        velocity_grid,
        energy_grid,
        levels=np.linspace(-4.0, 8.0, 25),
        colors="0.8",
        linewidths=0.8,
    )

    # The separatrix through the upright equilibrium.
    ax.contour(
        theta_grid,
        velocity_grid,
        energy_grid,
        levels=[0.0],
        colors="0.4",
        linestyles="--",
        linewidths=1.5,
    )

    # Touchdown guards: vertical lines that move with the control input.
    for alpha, style, label in [
        (ALPHA_MIN, "-", r"Guard $\theta = \alpha_{min} + \gamma$"),
        (ALPHA_MAX, "-.", r"Guard $\theta = \alpha_{max} + \gamma$"),
    ]:
        ax.axvline(
            alpha + incline,
            color="#c0392b",
            linestyle=style,
            linewidth=2,
            label=label,
        )

        ax.axvline(
            incline - alpha,
            color="#2e86c1",
            linestyle=style,
            linewidth=2,
            label=rf"Post-impact $\theta = \gamma - \alpha$ ($\alpha$ = {alpha:.3f})",
        )

    # The Poincare section.
    ax.axvline(
        0.0,
        color="#27ae60",
        linewidth=2.5,
        label=r"Poincaré section $\theta = 0$",
    )

    # Impact map arrows for both extreme actions.
    touchdown_velocity = 2.5

    for alpha in (ALPHA_MIN, ALPHA_MAX):
        touchdown_theta = alpha + incline

        post_theta = incline - alpha
        post_velocity = np.cos(2.0 * alpha) * touchdown_velocity

        ax.annotate(
            "",
            xy=(post_theta, post_velocity),
            xytext=(touchdown_theta, touchdown_velocity),
            arrowprops={
                "arrowstyle": "->",
                "color": "#8e44ad",
                "linewidth": 2,
            },
        )

    ax.plot(
        [],
        [],
        color="#8e44ad",
        linewidth=2,
        label="Impact reset map",
    )

    ax.plot(0.0, 0.0, "kx", markersize=10, label="Standing equilibrium")

    ax.set(
        xlim=(-theta_limit, theta_limit),
        ylim=(-velocity_limit, velocity_limit),
        xlabel=r"Angle $\theta$ (rad)",
        ylabel=r"Angular velocity $\dot{\theta}$ (rad/s)",
        title="Stance Phase Portrait, Guards and Impact Map",
    )

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    fig.savefig(output / "sketch_state_space.png", dpi=150)

    return fig


def run_sketch_analysis():
    """Generate the walker snapshots and the annotated state-space sketch."""

    output = Path("output/assignment_2")
    output.mkdir(parents=True, exist_ok=True)

    plot_walker_snapshots(params, output)
    plot_state_space_sketch(params, output)

    print(f"Saved {output / 'sketch_poses.png'}")
    print(f"Saved {output / 'sketch_state_space.png'}")

    plt.show()


# ============================================================
# Task 1: Walking simulation
# ============================================================

def run_walking_demo():
    """Run the passive walking simulation and save an animation."""

    initial_state = np.array([0.0, 3.0])
    initial_stance_position = np.array([0.0, 0.0])

    timestep = 1e-4
    sim_time = 3.0
    desired_number_of_steps = 3

    n_timesteps = round(sim_time / timestep) + 1

    time_traj = np.arange(n_timesteps) * timestep

    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    stance_traj = np.zeros((2, n_timesteps))
    stance_traj[:, 0] = initial_stance_position

    completed_steps = 0

    for step, t in enumerate(time_traj[:-1]):
        state = state_traj[:, step]
        stance_position = stance_traj[:, step]

        next_state = (
            state
            + timestep
            * model.dynamics(
                t,
                state,
                params,
            )
        )

        next_stance_position = stance_position.copy()

        if model.event_guard(
            state,
            next_state,
            params,
        ):
            next_stance_position = calculate_swing_foot_position(
                next_state,
                stance_position,
                params,
            )

            next_state = model.event_dynamics(
                next_state,
                params,
            )

            completed_steps += 1

        state_traj[:, step + 1] = next_state
        stance_traj[:, step + 1] = next_stance_position

        if completed_steps == desired_number_of_steps:
            break

    time_traj = time_traj[: step + 2]
    state_traj = state_traj[:, : step + 2]
    stance_traj = stance_traj[:, : step + 2]

    xmin = min(
        -0.5,
        np.min(stance_traj[0]) - 0.5,
    )

    xmax = np.max(stance_traj[0]) + 1.5
    ymin = np.min(stance_traj[1]) - 0.8
    ymax = np.max(stance_traj[1]) + 1.5

    view_limits = (
        xmin,
        xmax,
        ymin,
        ymax,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5),
        layout="constrained",
    )

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

    fps = 25
    frame_stride = round(
        1 / (fps * timestep)
    )

    frame_indices = list(
        range(
            0,
            time_traj.size,
            frame_stride,
        )
    )

    if frame_indices[-1] != time_traj.size - 1:
        frame_indices.append(
            time_traj.size - 1
        )

    animation = FuncAnimation(
        fig,
        draw_frame,
        frames=frame_indices,
        interval=1000 / fps,
        repeat=False,
    )

    output = Path(
        "output/assignment_2"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    animation.save(
        output / "walker.gif",
        writer=PillowWriter(fps=fps),
    )

    print(
        f"Saved {output / 'walker.gif'} "
        f"({completed_steps} footstrikes, "
        f"final x = "
        f"{stance_traj[0, -1]:.2f} m)."
    )

    plt.show()


# ============================================================
# Task 2: Standing controller
# ============================================================

def run_standing_controller_demo():
    """Test the standing controller near upright."""

    standing_initial_state = np.array([
        0.02,
        0.0,
    ])

    (
        standing_time,
        standing_states,
        standing_torques,
    ) = simulate_standing_controller(
        standing_initial_state,
        params,
        controller_params,
    )

    print("\nStanding controller test:")
    print(
        f"Initial state: "
        f"{standing_initial_state}"
    )
    print(
        f"Final state:   "
        f"{standing_states[:, -1]}"
    )
    print(
        f"Torque range: "
        f"[{standing_torques.min():.3f}, "
        f"{standing_torques.max():.3f}] N m"
    )

    output = Path(
        "output/assignment_2"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5),
        layout="constrained",
    )

    ax.plot(
        standing_time,
        standing_states[0],
        label=r"$\theta$",
    )

    ax.plot(
        standing_time,
        standing_states[1],
        label=r"$\dot{\theta}$",
    )

    ax.axhline(
        0.0,
        color="0.5",
        linestyle="--",
    )

    ax.set(
        xlabel="Time (s)",
        ylabel="State",
        title="Standing Controller Stabilization",
    )

    ax.legend()

    fig.savefig(
        output / "standing_controller.png",
        dpi=150,
    )

    print(
        f"Saved "
        f"{output / 'standing_controller.png'}"
    )

    plt.show()


# ============================================================
# Task 3: Region of Attraction
# ============================================================

def run_roa_analysis():
    """Compute and visualize the standing-controller RoA."""

    angle_values, velocity_values, roa = compute_roa(
        params,
        controller_params,
    )

    successful_states = np.count_nonzero(roa)
    total_states = roa.size

    print("\nRegion of Attraction:")
    print(
        f"Stable initial states: "
        f"{successful_states}/{total_states}"
    )

    print(
        f"Stable fraction: "
        f"{100 * successful_states / total_states:.1f}%"
    )

    # Check whether the chosen search window is too small.
    touches_velocity_boundary = (
        np.any(roa[0, :])
        or np.any(roa[-1, :])
    )

    print(
        f"RoA touches velocity search boundary: "
        f"{touches_velocity_boundary}"
    )

    output = Path("output/assignment_2")
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 6),
        layout="constrained",
    )

    image = ax.imshow(
        roa,
        origin="lower",
        aspect="auto",
        extent=[
            angle_values[0],
            angle_values[-1],
            velocity_values[0],
            velocity_values[-1],
        ],
        vmin=0,
        vmax=1,
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        ticks=[0, 1],
    )

    colorbar.ax.set_yticklabels([
        "Not stabilized",
        "Stabilized",
    ])

    ax.plot(
        0.0,
        0.0,
        "kx",
        markersize=8,
        label="Upright equilibrium",
    )

    ax.set(
        xlabel=r"Initial angle $\theta_0$ (rad)",
        ylabel=r"Initial angular velocity $\dot{\theta}_0$ (rad/s)",
        title="Standing Controller Region of Attraction",
    )

    ax.legend()

    fig.savefig(
        output / "standing_roa.png",
        dpi=150,
    )

    print(
        f"Saved {output / 'standing_roa.png'}"
    )

    plt.show()

# ============================================================
# Task 4: Poincare section
# ============================================================

def run_poincare_analysis():
    """Visualize the one-step Poincare map at theta = 0."""

    velocity_values = np.linspace(
        0.1,
        compute_maximum_velocity(params),
        100,
    )

    alpha_values = [
        ALPHA_MIN,
        0.5 * (ALPHA_MIN + ALPHA_MAX),
        ALPHA_MAX,
    ]

    output = Path("output/assignment_2")
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 6),
        layout="constrained",
    )

    for alpha in alpha_values:
        next_velocities = []

        for velocity in velocity_values:
            next_velocity = simulate_poincare_step(
                velocity,
                alpha,
                params,
            )

            if next_velocity is None:
                next_velocities.append(np.nan)
            else:
                next_velocities.append(next_velocity)

        next_velocities = np.array(
            next_velocities
        )

        ax.plot(
            velocity_values,
            next_velocities,
            label=(
                rf"$\alpha={alpha:.3f}$ rad"
            ),
        )

    ax.plot(
        velocity_values,
        velocity_values,
        "--",
        label="Identity",
    )

    ax.set(
        xlabel=r"$\dot{\theta}_k$ (rad/s)",
        ylabel=r"$\dot{\theta}_{k+1}$ (rad/s)",
        title=r"Poincaré Map at $\theta = 0$",
    )

    ax.legend()

    fig.savefig(
        output / "poincare_map.png",
        dpi=150,
    )

    print("\nPoincare section:")
    print("theta = 0 rad")
    print(
        f"Velocity range: "
        f"[{velocity_values[0]:.3f}, "
        f"{velocity_values[-1]:.3f}] rad/s"
    )
    print(
        "Alpha range: "
        f"[{ALPHA_MIN:.3f}, "
        f"{ALPHA_MAX:.3f}] rad"
    )

    # Simple numerical example.
    test_velocity = 2.0
    test_alpha = ALPHA_MIN

    next_velocity = simulate_poincare_step(
        test_velocity,
        test_alpha,
        params,
    )

    print(
        f"Example: omega_k = "
        f"{test_velocity:.3f} rad/s, "
        f"alpha = {test_alpha:.3f} rad"
    )

    print(
        f"         omega_(k+1) = "
        f"{next_velocity}"
    )

    print(
        f"Saved {output / 'poincare_map.png'}"
    )

    plt.show()

# ============================================================
# Task 5: State-action lookup table
# ============================================================

def run_lookup_table_analysis():
    """Build and visualize the Poincare state-action lookup table."""

    (
        velocity_values,
        alpha_values,
        transition_table,
    ) = build_transition_table(
        params,
    )

    valid_transitions = np.count_nonzero(
        np.isfinite(transition_table)
    )

    total_transitions = transition_table.size

    print("\nState-action lookup table:")
    print(
        f"Velocity states: {len(velocity_values)}"
    )
    print(
        f"Actions: {len(alpha_values)}"
    )
    print(
        f"Valid transitions: "
        f"{valid_transitions}/{total_transitions}"
    )

    print(
        f"Velocity range: "
        f"[{velocity_values[0]:.3f}, "
        f"{velocity_values[-1]:.3f}] rad/s"
    )

    print(
        f"Alpha range: "
        f"[{alpha_values[0]:.3f}, "
        f"{alpha_values[-1]:.3f}] rad"
    )

    output = Path("output/assignment_2")
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save locally so later tasks can reuse it.
    np.savez(
        output / "transition_table.npz",
        velocity_values=velocity_values,
        alpha_values=alpha_values,
        transition_table=transition_table,
    )

    fig, ax = plt.subplots(
        figsize=(8, 6),
        layout="constrained",
    )

    image = ax.imshow(
        transition_table,
        origin="lower",
        aspect="auto",
        extent=[
            velocity_values[0],
            velocity_values[-1],
            alpha_values[0],
            alpha_values[-1],
        ],
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        r"Next velocity $\dot{\theta}_{k+1}$ (rad/s)"
    )

    ax.set(
        xlabel=r"Current velocity $\dot{\theta}_k$ (rad/s)",
        ylabel=r"Angle of attack $\alpha_k$ (rad)",
        title="Poincaré State-Action Transition Table",
    )

    fig.savefig(
        output / "transition_table.png",
        dpi=150,
    )

    print(
        f"Saved {output / 'transition_table.png'}"
    )

    print(
        f"Saved {output / 'transition_table.npz'}"
    )

    plt.show()

# ============================================================
# Task 6: Minimum-step policy
# ============================================================

def run_policy_analysis():
    """Compute the minimum-step policy for reaching the RoA."""

    output = Path("output/assignment_2")

    table_file = output / "transition_table.npz"

    if not table_file.exists():
        raise FileNotFoundError(
            "Transition table not found. "
            "Run '--task lookup' first."
        )

    data = np.load(table_file)

    velocity_values = data["velocity_values"]
    alpha_values = data["alpha_values"]
    transition_table = data["transition_table"]

    roa_velocity_bound = compute_roa_velocity_bound(
        params,
        controller_params,
    )

    print(
        f"\nRoA boundary on the Poincaré section: "
        f"{roa_velocity_bound:.4f} rad/s"
    )

    minimum_steps, best_alpha = (
        compute_minimum_step_policy(
            velocity_values,
            alpha_values,
            transition_table,
            roa_velocity_bound,
        )
    )

    reachable = minimum_steps >= 0

    print("\nMinimum-step policy:")
    print(
        f"Reachable velocity states: "
        f"{np.count_nonzero(reachable)}"
        f"/{len(velocity_values)}"
    )

    if np.any(reachable):
        print(
            f"Maximum finite minimum steps: "
            f"{np.max(minimum_steps[reachable])}"
        )

    unique_steps = np.unique(
        minimum_steps[reachable]
    )

    for step_count in unique_steps:
        count = np.count_nonzero(
            minimum_steps == step_count
        )

        print(
            f"{step_count} step(s): "
            f"{count} states"
        )

    # Save numerical result for later trajectory analysis.
    np.savez(
        output / "minimum_step_policy.npz",
        velocity_values=velocity_values,
        minimum_steps=minimum_steps,
        best_alpha=best_alpha,
    )

    # --------------------------------------------------------
    # Plot minimum number of steps
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5),
        layout="constrained",
    )

    ax.plot(
        velocity_values[reachable],
        minimum_steps[reachable],
        "o-",
    )

    if np.any(~reachable):
        ax.plot(
            velocity_values[~reachable],
            np.zeros(
                np.count_nonzero(~reachable)
            ),
            "x",
            label="Not reachable",
        )

    ax.set(
        xlabel=(
            r"Initial Poincaré velocity "
            r"$\dot{\theta}_0$ (rad/s)"
        ),
        ylabel="Minimum walking steps to RoA",
        title="Minimum Steps to Reach Standing RoA",
    )

    ax.grid(True)
    ax.legend()

    fig.savefig(
        output / "minimum_steps.png",
        dpi=150,
    )

    # --------------------------------------------------------
    # Plot selected alpha policy
    # --------------------------------------------------------

    policy_valid = np.isfinite(best_alpha)

    fig_policy, ax_policy = plt.subplots(
        figsize=(8, 5),
        layout="constrained",
    )

    ax_policy.plot(
        velocity_values[policy_valid],
        best_alpha[policy_valid],
        "o-",
    )

    ax_policy.set(
        xlabel=(
            r"Poincaré velocity "
            r"$\dot{\theta}_k$ (rad/s)"
        ),
        ylabel=(
            r"Selected angle of attack "
            r"$\alpha_k$ (rad)"
        ),
        title="Minimum-Step Foot Placement Policy",
    )

    ax_policy.grid(True)

    fig_policy.savefig(
        output / "minimum_step_policy.png",
        dpi=150,
    )

    print(
        f"Saved "
        f"{output / 'minimum_steps.png'}"
    )

    print(
        f"Saved "
        f"{output / 'minimum_step_policy.png'}"
    )

    print(
        f"Saved "
        f"{output / 'minimum_step_policy.npz'}"
    )

    plt.show()


def load_policy_inputs(output):
    """Load the transition table and the minimum-step policy from disk."""
    table_file = output / "transition_table.npz"
    policy_file = output / "minimum_step_policy.npz"

    if not table_file.exists():
        raise FileNotFoundError("Run '--task lookup' first.")

    if not policy_file.exists():
        raise FileNotFoundError("Run '--task policy' first.")

    table_data = np.load(table_file)
    policy_data = np.load(policy_file)

    return (
        table_data["velocity_values"],
        table_data["alpha_values"],
        table_data["transition_table"],
        policy_data["minimum_steps"],
        policy_data["best_alpha"],
    )


def select_initial_index(minimum_steps, least_steps=3):
    """Pick the state that needs the most steps, among those needing enough."""
    candidates = np.where(minimum_steps >= least_steps)[0]

    if len(candidates) == 0:
        raise RuntimeError(
            f"No initial state requires at least {least_steps} steps."
        )

    return candidates[np.argmax(minimum_steps[candidates])]


def report_trajectory(name, trajectory, roa_velocity_bound):
    """Print what one closed-loop run actually did."""
    crossings = trajectory["states"][1, trajectory["poincare_indices"]]

    print(f"\n{name}:")
    print(f"  walking steps taken: {len(trajectory['actions'])}")
    print(f"  mid-stance velocities: {np.round(crossings, 4)}")
    print(f"  actions: {np.round(trajectory['actions'], 4)}")

    entry_state = trajectory["states"][:, trajectory["walking_end_index"]]
    final_state = trajectory["states"][:, -1]

    print(
        f"  state at RoA entry: {np.round(entry_state, 4)} "
        f"(bound {roa_velocity_bound:.4f})"
    )

    print(f"  final standing state: {np.round(final_state, 6)}")

    if np.max(np.abs(final_state)) > 1e-2:
        raise RuntimeError(
            f"{name} did not come to a standstill."
        )


def plot_discrete_trajectories(minimum, maximum, output):
    """Plot the velocity and action sequences of both closed-loop runs."""
    fig, (ax_velocity, ax_action) = plt.subplots(
        2,
        1,
        figsize=(8, 7),
        layout="constrained",
    )

    for trajectory, label, style in [
        (minimum, "Minimum-step policy", "o-"),
        (maximum, "Maximum-step policy", "s--"),
    ]:
        crossings = trajectory["states"][1, trajectory["poincare_indices"]]

        ax_velocity.plot(
            np.arange(len(crossings)),
            crossings,
            style,
            label=label,
        )

        ax_action.step(
            np.arange(len(trajectory["actions"])),
            trajectory["actions"],
            where="post",
            label=label,
        )

    ax_velocity.set(
        xlabel="Mid-stance crossing index",
        ylabel=r"Angular velocity $\dot{\theta}$ (rad/s)",
        title="Closed-Loop Velocity and Action Sequences",
    )

    ax_velocity.grid(True)
    ax_velocity.legend()

    ax_action.set(
        xlabel="Walking step",
        ylabel=r"Angle of attack $\alpha$ (rad)",
    )

    ax_action.grid(True)
    ax_action.legend()

    fig.savefig(output / "step_trajectories.png", dpi=150)

    return fig


def plot_continuous_trajectories(minimum, maximum, output):
    """Plot both full state-space trajectories, from walking to standstill."""
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")

    for trajectory, label, marker in [
        (minimum, "Minimum-step", "o"),
        (maximum, "Maximum-step", "s"),
    ]:
        states = trajectory["states"]

        line, = ax.plot(
            states[0],
            states[1],
            linewidth=1.0,
            label=f"{label} trajectory",
        )

        ax.plot(
            states[0, trajectory["poincare_indices"]],
            states[1, trajectory["poincare_indices"]],
            marker,
            color=line.get_color(),
            label=f"{label} mid-stance crossings",
        )

        ax.plot(
            states[0, trajectory["impact_indices"]],
            states[1, trajectory["impact_indices"]],
            "v",
            color=line.get_color(),
            markersize=5,
            label=f"{label} impacts",
        )

    ax.axvline(
        0.0,
        color="#27ae60",
        linewidth=1.5,
        label=r"Poincaré section $\theta = 0$",
    )

    ax.plot(0.0, 0.0, "kx", markersize=10, label="Standing equilibrium")

    ax.set(
        xlabel=r"Angle $\theta$ (rad)",
        ylabel=r"Angular velocity $\dot{\theta}$ (rad/s)",
        title="Continuous State-Space Trajectories",
    )

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.savefig(output / "continuous_trajectories.png", dpi=150)

    return fig


# ============================================================
# Task 7: Minimum- and maximum-step trajectories
# ============================================================

def run_trajectory_analysis():
    """Compare the shortest and longest walks to a standstill."""

    output = Path("output/assignment_2")

    (
        velocity_values,
        alpha_values,
        transition_table,
        minimum_steps,
        minimum_alpha,
    ) = load_policy_inputs(output)

    roa_velocity_bound = compute_roa_velocity_bound(
        params,
        controller_params,
    )

    maximum_steps, maximum_alpha = compute_maximum_step_policy(
        velocity_values,
        alpha_values,
        transition_table,
        roa_velocity_bound,
    )

    initial_index = select_initial_index(minimum_steps)
    initial_velocity = velocity_values[initial_index]

    print("\nTrajectory analysis:")
    print(f"Initial velocity: {initial_velocity:.4f} rad/s")
    print(f"Tabulated minimum steps: {minimum_steps[initial_index]}")
    print(f"Tabulated maximum steps: {maximum_steps[initial_index]}")

    minimum = simulate_policy_trajectory(
        initial_velocity,
        velocity_values,
        minimum_alpha,
        params,
        controller_params,
        roa_velocity_bound,
    )

    maximum = simulate_policy_trajectory(
        initial_velocity,
        velocity_values,
        maximum_alpha,
        params,
        controller_params,
        roa_velocity_bound,
    )

    report_trajectory("Minimum-step policy", minimum, roa_velocity_bound)
    report_trajectory("Maximum-step policy", maximum, roa_velocity_bound)

    plot_discrete_trajectories(minimum, maximum, output)
    plot_continuous_trajectories(minimum, maximum, output)

    np.savez(
        output / "trajectory_analysis.npz",
        initial_velocity=initial_velocity,
        minimum_actions=minimum["actions"],
        maximum_actions=maximum["actions"],
        minimum_crossings=minimum["states"][1, minimum["poincare_indices"]],
        maximum_crossings=maximum["states"][1, maximum["poincare_indices"]],
    )

    print(f"\nSaved {output / 'step_trajectories.png'}")
    print(f"Saved {output / 'continuous_trajectories.png'}")
    print(f"Saved {output / 'trajectory_analysis.npz'}")

    plt.show()


# ============================================================
# Command-line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "MAE5110 Assignment 2 experiments"
        )
    )

    parser.add_argument(
        "--task",
        required=True,
    choices=[
        "sketches",
        "walking",
        "standing",
        "roa",
        "poincare",
        "lookup",
        "policy",
        "trajectory",
        "grid",
    ],
        help="Select which experiment to run.",
    )

    args = parser.parse_args()

    if args.task == "sketches":
        run_sketch_analysis()

    elif args.task == "walking":
        run_walking_demo()

    elif args.task == "standing":
        run_standing_controller_demo()

    elif args.task == "roa":
        run_roa_analysis()

    elif args.task == "poincare":
        run_poincare_analysis()

    elif args.task == "lookup":
        run_lookup_table_analysis()

    elif args.task == "policy":
        run_policy_analysis()

    elif args.task == "trajectory":
        run_trajectory_analysis()

    elif args.task == "grid":
        run_grid_analysis()

if __name__ == "__main__":
    main()