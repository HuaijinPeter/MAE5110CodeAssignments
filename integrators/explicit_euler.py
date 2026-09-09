
def step(dynamics, t, state, timestep, params):
    next_state = state + timestep * dynamics(t, state, params)
    return next_state