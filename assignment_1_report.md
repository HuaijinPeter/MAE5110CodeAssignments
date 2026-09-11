# Assignment 1: Rimless-Wheel Dynamics and Stability

## Reproducing the results

From the repository root, install the environment once:

```console
uv sync --python 3.14
```

Run the sanity checks:+

```console
uv run python assignment_1_sanity_check.py+
```

Run each experiment with the following copy/paste commands:

```console
uv run python assignment_1.py --experiment initial
uv run python assignment_1.py --experiment roa
uv run python assignment_1.py --experiment return_map
uv run python assignment_1.py --experiment slope_sweep
uv run python assignment_1.py --experiment spoke_sweep
```

The RoA and parameter sweeps are brute-force calculations and take longer than
the initial-condition and return-map experiments. Figures and numerical data
are saved under `results/assignment_1/`.

## Model implementation

The wheel has $N$ massless spokes of length $l$, with a point mass at the
hub. The half-angle between adjacent spokes is

$$
\alpha = \frac{\pi}{N}.
$$

The state is $x=[\theta,\dot\theta]$, where $\theta$ is measured from the
upward vertical and is positive downhill. While one spoke is in contact with
the ground, the wheel follows the inverted-pendulum dynamics

$$
\dot x =
\begin{bmatrix}
\dot\theta\\
\dfrac{g}{l}\sin\theta
\end{bmatrix}.
$$

At a forward impact,

$$
\theta^- = \gamma+\alpha,
\qquad
\theta^+ = \gamma-\alpha,
\qquad
\dot\theta^+ = \dot\theta^-\cos(2\alpha).
$$

The implementation also includes the corresponding backward impact so that
initial conditions with negative angular velocity can be classified. The
default parameters used for the main experiments are:

| Parameter | Value |
|---|---:|
| Gravity, $g$ | $9.81\ \mathrm{m/s^2}$ |
| Spoke length, $l$ | $1.0\ \mathrm{m}$ |
| Number of spokes, $N$ | 8 |
| Slope, $\gamma$ | $5^\circ$ |
| Half inter-spoke angle, $\alpha$ | $22.5^\circ$ |

## Sanity checks

The sanity checks separately test the geometry, continuous dynamics, event
guards, and collision resets.

| Check | Expected result | Observed result |
|---|---|---|
| Geometry for $N=8$ | $\alpha=\pi/8=22.5^\circ$ | $22.5^\circ$ |
| Dynamics at $[\theta,\dot\theta]=[0.1,0]$ | $[0,9.81\sin(0.1)]\approx[0,0.97937]$ | $[0,0.97937]$ |
| Forward guard below contact | No impact at $20^\circ$ | `False` |
| Forward guard at contact | Impact at $\gamma+\alpha=27.5^\circ$ | `True` |
| Forward reset from $\dot\theta^-=2$ | $\theta^+=-17.5^\circ$, $\dot\theta^+=2\cos45^\circ=1.41421$ | $-17.5^\circ,1.41421$ |
| Backward reset from $\dot\theta^-=-2$ | $\theta^+=27.5^\circ$, $\dot\theta^+=-1.41421$ | $27.5^\circ,-1.41421$ |

All checks matched their expected values. In particular, the velocity retains
its direction and is reduced by the plastic-impact factor
$\cos(2\alpha)$.

## Regions of attraction

I sampled a $41\times41$ grid over

$$
\theta\in[\gamma-\alpha,\gamma+\alpha],
\qquad
\dot\theta\in[-3,3]\ \mathrm{rad/s}.
$$

Each initial condition was integrated with RK4 using
$\Delta t=10^{-3}\ \mathrm{s}$, for at most 40 seconds. A trajectory that
completed 20 consecutive forward impacts was classified as converging to the
rolling limit cycle. If its post-impact energy could no longer carry the hub
over the upright configuration, the dissipative rocking sequence was
classified as converging to the stable two-spoke resting configuration. A
third label records trajectories that cannot be classified within the time
limit.

![Regions of attraction](results/assignment_1/roa/roa.png)

For the default parameters, the sampled basin fractions were:

| Attractor | Fraction |
|---|---:|
| Resting fixed point | 0.51695 |
| Rolling limit cycle | 0.48305 |
| Unresolved | 0.00000 |

The square markers show the two reduced-coordinate representations of the
same resting contact configuration. The white dashed curve is the rolling
hybrid limit cycle. For positive initial velocity, the numerical boundary
agrees with the black energy-based boundary. Negative initial velocities
produce several bands because the wheel may take different numbers of uphill
steps before collision losses either trap it at rest or allow it to reverse
and approach the downhill rolling cycle.

## Poincaré return map and Floquet multiplier

I used the post-impact contact state
$\theta=\gamma-\alpha$ as the Poincaré section and evaluated one forward
step for 151 initial angular velocities between 0.5 and 2.0 rad/s. The return
map used $\Delta t=10^{-4}\ \mathrm{s}$.

![One-dimensional Poincaré return map](results/assignment_1/return_map/return_map.png)

The intersection with the identity line gives

$$
\dot\theta^* = 1.14422\ \mathrm{rad/s}.
$$

As a consistency check, conservation of energy during a swing and the impact
reset give a theoretical fixed point of $1.14402\ \mathrm{rad/s}$, an
absolute difference of approximately $2.1\times10^{-4}\ \mathrm{rad/s}$.

I estimated the local return-map slope by perturbing the fixed point on both
sides by $0.01\ \mathrm{rad/s}$ and using
$\Delta t=10^{-5}\ \mathrm{s}$:

$$
\lambda \approx
\frac{P(\dot\theta^*+0.01)-P(\dot\theta^*-0.01)}{0.02}
=0.49929.
$$

The theoretical multiplier is

$$
\lambda_{\mathrm{theory}}=\cos^2(2\alpha)=0.5.
$$

Because $|\lambda|<1$, the rolling fixed point of the return map, and hence
the rolling limit cycle, is locally stable.

## Effect of slope

I swept the slope from $2^\circ$ through $8^\circ$, keeping $N=8$.

| Slope | Rolling RoA | Resting RoA | Unresolved | Fixed-point speed (rad/s) | Floquet multiplier |
|---:|---:|---:|---:|---:|---:|
| $2^\circ$ | 0.0000 | 1.0000 | 0.0000 | none | none |
| $3^\circ$ | 0.0000 | 0.9990 | 0.0010 | none | none |
| $4^\circ$ | 0.4037 | 0.5963 | 0.0000 | 1.0239 | 0.5008 |
| $5^\circ$ | 0.4870 | 0.5130 | 0.0000 | 1.1442 | 0.4990 |
| $6^\circ$ | 0.5536 | 0.4454 | 0.0010 | 1.2531 | 0.5009 |
| $7^\circ$ | 0.6098 | 0.3902 | 0.0000 | 1.3533 | 0.5006 |
| $8^\circ$ | 0.6681 | 0.3319 | 0.0000 | 1.4459 | 0.4994 |

| Regions of attraction | Local convergence |
|---|---|
| ![Slope versus RoA](results/assignment_1/slope_sweep/slope_vs_roa.png) | ![Slope versus Floquet multiplier](results/assignment_1/slope_sweep/slope_vs_floquet.png) |

For this wheel, the theoretical onset of a viable rolling cycle is
approximately $3.91^\circ$. This explains why no rolling cycle was found at
$2^\circ$ or $3^\circ$, while one appears at $4^\circ$. Increasing slope
adds more gravitational energy per step, so the rolling basin grows and the
resting basin shrinks. The rolling fixed-point velocity also increases.

The Floquet multiplier stays near 0.5 because, for fixed $N$, its theoretical
value $\cos^2(2\alpha)$ does not depend on slope. The maximum numerical
deviation from theory in the viable cases was less than $9.9\times10^{-4}$.
The very small unresolved fractions at $3^\circ$ and $6^\circ$ each
correspond to one grid point at the unstable upright equilibrium
$(\theta,\dot\theta)=(0,0)$, rather than a stable attractor.

## Effect of the number of spokes

I swept all integer spoke counts from 6 through 12 at a fixed $5^\circ$
slope.

| Spokes | Rolling RoA | Resting RoA | Fixed-point speed (rad/s) | Numerical / theoretical Floquet |
|---:|---:|---:|---:|---:|
| 6 | 0.0000 | 1.0000 | none | none |
| 7 | 0.0000 | 1.0000 | none | none |
| 8 | 0.4870 | 0.5130 | 1.1445 | 0.5000 / 0.5000 |
| 9 | 0.6202 | 0.3798 | 1.2894 | 0.5858 / 0.5868 |
| 10 | 0.7399 | 0.2601 | 1.4151 | 0.6560 / 0.6545 |
| 11 | 0.7867 | 0.2133 | 1.5276 | 0.7079 / 0.7077 |
| 12 | 0.8127 | 0.1873 | 1.6305 | 0.7499 / 0.7500 |

| Regions of attraction | Local convergence |
|---|---|
| ![Spoke count versus RoA](results/assignment_1/spoke_sweep/spokes_vs_roa.png) | ![Spoke count versus Floquet multiplier](results/assignment_1/spoke_sweep/spokes_vs_floquet.png) |

At a $5^\circ$ slope, wheels with 6 or 7 spokes do not have enough
post-impact energy to sustain the rolling cycle. The viable cycle first
appears at $N=8$. As $N$ increases, the angle between spokes and the
energy loss at each collision decrease. Consequently, the rolling RoA grows
from 0.4870 at $N=8$ to 0.8127 at $N=12$.

At the same time, the multiplier increases toward one, from approximately 0.5
to 0.75. Therefore, more spokes make sustained rolling accessible from more
initial conditions but make local convergence to the final cycle slower. The
numerical multipliers closely follow $\cos^2(2\pi/N)$; the maximum absolute
error was approximately $1.5\times10^{-3}$.

## Conclusions

The default rimless wheel has two stable long-term behaviors: a resting
two-spoke contact configuration and a downhill rolling limit cycle. Their
basins cover all sampled initial conditions. A steeper slope or a larger
number of spokes enlarges the rolling basin. Slope changes the cycle speed but
has essentially no effect on its local multiplier for fixed $N$. Increasing
the number of spokes reduces collision losses and enlarges the rolling basin,
but it also moves the Floquet multiplier closer to one and slows local
convergence.
