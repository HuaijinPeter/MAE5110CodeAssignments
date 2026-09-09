import numpy as np


def dynamics(t, state, params):
    gravity = params["gravity"]

    # state : [height, velocity]
    height = state[0]
    velocity = state[1]

    # the change in height is the velocity, the change in velocity is the acceleration due to gravity
    state_derivative = np.array([
        velocity,
        -gravity,
    ])

    return state_derivative



def handle_impact(state, params):
    restitution = params["restitution"]

    height = state[0]
    velocity = state[1]

    if height < 0 and velocity < 0:
        height = 0.0

        # the velocity loss, and the direction change
        velocity = -restitution * velocity

    return np.array([height, velocity])


def calculate_energy(state, params):
    gravity = params["gravity"]
    mass = params["mass"]

    height = state[0]
    velocity = state[1]

    kinetic_energy = 0.5 * mass * velocity**2
    potential_energy = mass * gravity * height

    return kinetic_energy, potential_energy
