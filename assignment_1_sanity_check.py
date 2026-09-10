import numpy as np

from models import rimless_wheel as model

# para test
params = model.generate_params()
alpha = model.calculate_alpha(params)

print("alpha =", alpha)
print("alpha in degrees =", np.rad2deg(alpha))


# state derivative test
state = np.array([0.1, 0.0])
state_derivative = model.dynamics(0.0, state, params)
print("state derivative =", state_derivative)


# impact impact
state_before_impact = np.array([
    np.deg2rad(20.0),
    1.0,
])

print(model.detect_impact(state_before_impact, params))

state_at_impact = np.array([
    np.deg2rad(27.5),
    1.0,
])

print(model.detect_impact(state_at_impact, params))




# test the application
state_before_impact = np.array([
    np.deg2rad(27.5),
    2.0,
])

state_after_impact = model.apply_impact(
    state_before_impact,
    params,
)

print(
    "new angle =",
    np.rad2deg(state_after_impact[0]),
)

print(
    "new angular velocity =",
    state_after_impact[1],
)