import numpy as np

def generate_params():
    params = {
        "gravity": 9.81,  # m/s^2
        "length": 1.0,  # m
        "num_spokes": 8,  # number of spokes
        "slope": np.deg2rad(5.0),# convert to radians
    }
    return params


def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = gravity / length * np.sin(angle)

    state_derivative = np.array([
        angular_velocity,
        angular_acceleration,
    ])

    return state_derivative

def calculate_alpha(params):
    num_spokes = params["num_spokes"]
    alpha = np.pi / num_spokes
    return alpha


def detect_impact(state, params):
    """Detect a forward (downhill) spoke impact."""
    angle = state[0]
    angular_velocity = state[1]
    slope = params["slope"]
    alpha = calculate_alpha(params)
    impact_angle = slope + alpha

    return angle >= impact_angle and angular_velocity > 0


def detect_backward_impact(state, params):
    """Detect a backward (uphill) spoke impact."""
    angle = state[0]
    angular_velocity = state[1]
    slope = params["slope"]
    alpha = calculate_alpha(params)
    impact_angle = slope - alpha

    return angle <= impact_angle and angular_velocity < 0



def apply_impact(state, params):
    """Apply the forward-impact coordinate and velocity reset."""
    angular_velocity = state[1]

    slope = params["slope"]
    alpha = calculate_alpha(params)

    new_angle = slope - alpha
    new_angular_velocity = angular_velocity * np.cos(2 * alpha)

    new_state = np.array([
        new_angle,
        new_angular_velocity,
    ])

    return new_state


def apply_backward_impact(state, params):
    """Apply the backward-impact coordinate and velocity reset."""
    angular_velocity = state[1]

    slope = params["slope"]
    alpha = calculate_alpha(params)

    new_angle = slope + alpha
    new_angular_velocity = angular_velocity * np.cos(2 * alpha)

    new_state = np.array([
        new_angle,
        new_angular_velocity,
    ])

    return new_state
