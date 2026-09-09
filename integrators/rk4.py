# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods

def step(dynamics, t, state, timestep, params):

    k1 = dynamics(t, state, params)

    k2 = dynamics(
        t + timestep / 2,
        state + timestep * k1 / 2,
        params,
    )

    k3 = dynamics(
        t + timestep / 2,
        state + timestep * k2 / 2,
        params,
    )

    k4 = dynamics(
        t + timestep,
        state + timestep * k3,
        params,
    )

    next_state = state + timestep / 6 * (
        k1 + 2 * k2 + 2 * k3 + k4
    )

    return next_state