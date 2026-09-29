import numpy as np

from integrators import rk4 as integrator
from models import pendulum as model


def test_energy_conservation():
    params = model.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0

    state = np.array([0.5, 0.0])
    timestep = 0.001

    total_energies = []

    for step in range(101):
        kinetic_energy, potential_energy = model.calculate_energy(state, params)
        total_energies.append(kinetic_energy + potential_energy)

        if step < 100:
            state = integrator(
                model.dynamics,
                step * timestep,
                state,
                timestep,
                params,
            )

    energy_changes = np.diff(total_energies)

    assert np.all(np.isclose(energy_changes, 0.0, atol=1e-8))


def test_positive_torque_produces_positive_acceleration():
    params = model.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 1.0

    state = np.array([0.0, 0.0])
    derivative = model.dynamics(0.0, state, params)

    expected_acceleration = (
        params["torque"] / (params["mass"] * params["length"] ** 2)
    )

    assert np.isclose(derivative[1], expected_acceleration)


def test_damping_opposes_motion():
    params = model.generate_params()
    params["torque"] = 0.0
    params["damping_coeff"] = 0.5

    state = np.array([0.0, 1.0])
    derivative = model.dynamics(0.0, state, params)

    expected_acceleration = (
        -params["damping_coeff"]
        * state[1]
        / (params["mass"] * params["length"] ** 2)
    )

    assert np.isclose(derivative[1], expected_acceleration)