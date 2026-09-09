# Assignment 1

**Due date:** Wednesday, September 9th, 11:59PM (midnight)

**Goals:**

- get more comfortable with git;
- analyze dynamics using mathematical and numerical tools; and
- model and analyze hybrid dynamics.

## Coding Practices

### Git

#### Fork the repo

If you haven't yet, set up your own copy of the repository on GitHub. Convention is to call your fork `origin` and original you forked from `upstream`. So let's do that.

First, go to the [class repository](https://github.com/LampLighterLab/MAE5110CodeAssignments) in your browser and click the `Fork` button. Since you originally directly cloned the class repository, your local copy calls it `origin`, and is not aware of your fork. You can check this by running

```console
git remote -v
```
Let's rename the class repository, and then add your new fork.
```console
git remote rename origin upstream
```

Next, add your fork as `origin`, replacing the URL below with the URL of your fork:

```console
git remote add origin <your-fork-url>
git push -u origin main
```

Check your work with:

```console
git remote -v
```

You should see your fork listed as `origin` and the class repository listed as `upstream`. From now on, push your own progress to `origin` (this should be the default if you `git push`) and get new assignments and other course material from `upstream`.

To bring new material from the class repository into your `main` branch, run each of the following commands and resolve any merge conflicts that arise:

```console
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

#### Branches and Git Workflow

Also, start getting into the habit of using branches to shape your workflow.
Your `main` branch should _always_ work and be bug-free. If you want to add new functionality, you risk breaking things; so instead of doing it on the `main` branch, first create a new branch:

```console
git branch <netID>/assignment_1
```

It is common convention to use your GitHub username or other identifier preceding the topic of the branch, for this class, use your netID.
Then switch to that branch:

```console
git switch <netID>/assignment_1
```

While you're on this branch, you should use the branch name to remind yourself to focus on _this_ topic. If you find you need to work on something else for a while, that's fine. Save your progress: either make a commit, or simply stash with [`git stash`](https://www.atlassian.com/git/tutorials/saving-changes/git-stash).
Then you can use `git switch main` to go back to `main`, or create another branch to work on something else.

At what granularity should you create new branches? Similarly to commit granularity, this is largely a matter of style and experience; try it out and see what works for you.

### Pull Requests

Once you're done, open a _pull request_ on GitHub from your branch into your main branch. Starting next week, we will also assign peer code reviews.

### Clean Code of the week: naming variables and functions

Try to write code that can be read as easily as a Steinbeck novel. It should be concise, use simple English and simple structure, and be to the point. Remember, clarity is king.  Follow these guidelines:

- names should indicate purpose.
  - _bad_: `m`
  - _good_: `mass` or `pendulum_mass`. Consider your use-case if the extra specification is useful.
- function names should start with a verb that describes what will happen.
  - _bad_: `def event_guard(self)` 
  - _good_: `def detect_event(self)`.
- Use pronounceable names: we're better at remembering things we can say
  - _bad_: `dPdx`
  - _good_: `poincare_map_jacobian`

## Model: the Rimless Wheel

This week's model, both in class and in this assignment, is going to be the _rimless wheel_.
Imagine it as a wagon wheel, but the rim has been removed, so it clunkily rolls on its spokes.
We will model the wheel as a point mass at its hub with evenly spaced, massless spokes of length $l$. The stance spoke is pinned to the ground and does not slip: it will have the same dynamics as the (inverted) pendulum.
Let $N$ be the number of spokes and let $2\alpha = 2\pi/N$ be the angle between adjacent spokes. Let $\gamma$ be the downhill inclination of the ground and $\theta$ be the angle of the stance spoke, measured from the upward vertical and positive in the downhill direction. The state is $[\theta, \dot\theta]$.
Before starting to code this model, sketch it out, with parameters and states annotated.

Like the bouncing ball, the rimless wheel has impact events and experiences nonsmooth jumps in the dynamics! Unlike the bouncing ball, we will model these as an instantaneous plastic collision: at the contact event, the stance spoke is instantly switched to the new spoke (including a coordinate shift!), and the new velocity should be calculated such that angular momentum about the new contact point is conserved, as covered in class: $\dot{\theta}^+ = \dot{\theta}^- \cos(2 \alpha)$.

Your first task is to implement the rimless-wheel continuous dynamics, impact-event guard, and reset dynamics.
As usual, before you implement the code, start with defining a sanity check or two: what experiment will you run to make sure the model is behaving correctly? What do you expect to see?

## Analysis

Now you're going to do some stability analysis.
We will focus on the slope incline $\gamma$. You may pick all other parameters (except gravity, use Earth gravity).

First, estimate the _regions of attraction_ (RoA) by brute force. Create a grid over the state space and simulate the system from every grid point, long enough to be sure it has reached steady-state.
Classify each attractor (how many do you expect?), then plot the results in the state space as a map of which attractor each grid-point converges to.
_Hint:_ this might be slow, especially since (as you should have noticed from assignment 0) the non-smooth impact requires much smaller timesteps to be precise than the pendulum swing. What can you do to make this faster, but still accurate?

Use your contact event as a Poincaré section and construct a one-dimensional step-to-step return map. Plot the angular velocity at one crossing against the angular velocity at the next crossing, together with the identity line. Use this plot to identify the fixed point of the return map.

Estimate the Floquet multiplier of the rolling limit cycle. Perturb the post-impact angular velocity on both sides of the return-map fixed point and estimate its local slope.

Finally, sweep the inclinations to see how it affects both the RoA and the Floquet multiplier. Then do the same sweeping number of spokes between 6 and 12. Visualize and explain your findings.

### Deliverables

Push your assignment branch to your fork with `git push -u origin <your-branch-name>`. Then open a pull request (PR) from that branch into the `main` branch of your own fork, and submit the link to that PR on Canvas. The PR should contain:

- your rimless-wheel implementation (in code), with clear copy/paste instructions for running it; **(5 pts)**

A markdown file reporting:
- an explanation of your sanity checks, including what you expected and what happened; **(5 pts)**
- a state-space plot showing the RoA of every stable attractor, including fixed points and limit cycles; **(5 pts)**
- your one-dimensional return-map plot, with its fixed point and the identity line clearly marked; and **(5 pts)**
- visualization and discussion of how the slope and number of spokes affects the RoA and local convergence **(10 pts)**
