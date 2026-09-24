"""Checks for the inverted-pendulum-walker model and its controllers.

Run with `uv run pytest test_walker.py`.
"""

import numpy as np
import pytest

import assignment_2 as experiment
from models import inverted_pendulum_walker as model

params = model.generate_params()


def compute_return_map_from_energy(initial_velocity, angle_of_attack, params):
    """
    Return the next theta = 0 crossing velocity in closed form.

    With no ankle torque the stance phase conserves energy, so each arc
    can be solved exactly. This is the oracle the simulated Poincaré map
    is checked against. Returns None when the walker stalls on an arc.
    """
    gravity = params["gravity"]
    length = params["length"]
    incline = params["incline"]

    touchdown_angle = angle_of_attack + incline
    post_impact_angle = incline - angle_of_attack

    touchdown_velocity_squared = initial_velocity**2 + 2 * (gravity / length) * (
        1.0 - np.cos(touchdown_angle)
    )

    if touchdown_velocity_squared <= 0.0:
        return None

    post_impact_velocity = np.cos(2.0 * angle_of_attack) * np.sqrt(
        touchdown_velocity_squared
    )

    next_velocity_squared = post_impact_velocity**2 + 2 * (gravity / length) * (
        np.cos(post_impact_angle) - 1.0
    )

    if next_velocity_squared <= 0.0:
        return None

    return np.sqrt(next_velocity_squared)


@pytest.mark.parametrize("initial_velocity", [1.0, 2.0, 3.0, 4.0, 4.4])
@pytest.mark.parametrize(
    "angle_of_attack", [experiment.ALPHA_MIN, experiment.ALPHA_MAX]
)
def test_simulated_return_map_matches_energy_solution(
    initial_velocity, angle_of_attack
):
    """The integrated Poincaré map must agree with the exact solution."""
    simulated = experiment.simulate_poincare_step(
        initial_velocity,
        angle_of_attack,
        params,
    )

    exact = compute_return_map_from_energy(
        initial_velocity,
        angle_of_attack,
        params,
    )

    if exact is None:
        assert simulated is None
        return

    assert simulated is not None

    # Well below the coarsest grid spacing used for the look-up table.
    assert abs(simulated - exact) < 1e-2


@pytest.mark.parametrize(
    "angle_of_attack", [experiment.ALPHA_MIN, experiment.ALPHA_MAX]
)
def test_impact_dissipates_energy(angle_of_attack):
    """The collision map must never add energy."""
    impact_params = params.copy()
    impact_params["angle_of_attack"] = angle_of_attack

    touchdown_state = np.array([angle_of_attack + params["incline"], 2.5])

    post_impact_state = model.event_dynamics(touchdown_state, impact_params)

    energy_before = model.calculate_energy(touchdown_state, impact_params)
    energy_after = model.calculate_energy(post_impact_state, impact_params)

    assert energy_after <= energy_before


def test_event_guard_fires_once_at_the_touchdown_angle():
    """The guard must trigger on the crossing of theta = alpha + gamma."""
    touchdown_angle = params["angle_of_attack"] + params["incline"]

    before = np.array([touchdown_angle - 1e-3, 1.0])
    after = np.array([touchdown_angle + 1e-3, 1.0])

    assert model.event_guard(before, after, params)
    assert not model.event_guard(before, before, params)
    assert not model.event_guard(after, after, params)


def test_ankle_torque_respects_the_admissible_bounds():
    """The controller must saturate inside the assignment's torque limits."""
    torque_min = (
        experiment.TORQUE_MIN_FRACTION
        * params["mass"]
        * params["gravity"]
        * params["length"]
    )

    torque_max = (
        experiment.TORQUE_MAX_FRACTION
        * params["mass"]
        * params["gravity"]
        * params["length"]
    )

    for theta in np.linspace(-np.pi / 2, np.pi / 2, 41):
        for angular_velocity in np.linspace(-5.0, 5.0, 41):
            torque = experiment.compute_ankle_torque(
                np.array([theta, angular_velocity]),
                params,
                experiment.controller_params,
            )

            assert torque_min - 1e-12 <= torque <= torque_max + 1e-12


def test_feedback_linearization_cancels_gravity_when_unsaturated():
    """Inside the linear range the closed loop must behave like -kp*x - kd*v."""
    kp = experiment.controller_params["kp"]
    kd = experiment.controller_params["kd"]

    state = np.array([0.01, 0.01])

    closed_loop_params = params.copy()
    closed_loop_params["ankle_torque"] = experiment.compute_ankle_torque(
        state,
        params,
        experiment.controller_params,
    )

    angular_acceleration = model.dynamics(0.0, state, closed_loop_params)[1]

    expected = -kp * state[0] - kd * state[1]

    assert angular_acceleration == pytest.approx(expected, abs=1e-9)


def test_standing_controller_catches_states_inside_the_roa():
    """States below the measured RoA boundary must come to a standstill."""
    assert experiment.is_stabilized(
        np.array([0.0, 0.2]),
        params,
        experiment.controller_params,
    )

    assert not experiment.is_stabilized(
        np.array([0.0, 1.2]),
        params,
        experiment.controller_params,
    )
