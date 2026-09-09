# Assignment 0

Assignment 0 is pass/fail (it's not graded, but you will be asked to drop the class if you do not do it), and the learning objectives are for you to familiarize yourself with the following tools:

- `git`
- `uv`
- VS Code, Python debugger

as well as some basic debugging habits and coding style.

## Development Environment

Linux and the command line interface (CLI) is standard in robotics, and you'll be expected to be comfortable with it. If your laptop runs...

**Linux** (typically Ubuntu): you should be good to go.

**macOS**: you have the same `unix` system under the hood and that will work for the 2D simulations; make sure you have the [XCode Command Line Tools](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools) installed. I also recommend getting the iTerm2 terminal.

**Windows**: start by setting up [the Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install).

## Git ready

If you don't already have it, install [git](https://github.com/git-guides).
We will use git a lot. You can use it in the command line interface (CLI) or with a graphical user interface (GUI).
Especially if you aren't very familiar with git, I strongly recommend you to start with the CLI; it's much simpler and will force you to understand what you're doing.

Once you are in the CLI, you need to tell `git` about who you are. This means that you need to run the following:

```bash
git config --global user.name "Your Name"
git config --global user.email "Your Email"
```

Once you're ready, clone this repo.

## Set up your UV environment.

In this class, we will use [uv](https://docs.astral.sh/uv/) as our environment manager. You should install it, then run `uv sync --python 3.14` in this folder to set up your project.

If you're not familiar with environment managers, or why they are important, see [here](https://xkcd.com/1987/), and in all seriousness, read the _Why_ and _How_ sections [here](https://realpython.com/python-virtual-environments-a-primer/?utm_source=chatgpt.com#why-do-you-need-virtual-environments).

## Pendulum simulation
_Before_ running the code: read through the pendulum simulation in `assignment_0.py`, and predict what you expect to happen. Then run it.

Set up the [Python debugger in VS Code](https://code.visualstudio.com/docs/python/debugging), and step through the code, using the debugger to follow the codeflow into different parts of the code. Yes, this codebase is tiny and you don't really need to do this to understand where things are implemented; in more complex code-bases (e.g. the RL codebase you will work with in this class), following through the codeflow once with the debugger is a good habit to have.

Sketch the pendulum, including coordinate system, and use it to verify your intuition/predictions of the numerical results. Bring your sketch to class on [Friday/Monday].

## Create a module for numerical integrators.

In `assignment_0.py`, the integration step is currently written out explicitly in the script's simulation loop with a simple Explicit Euler method. This integration method works, but is very sensitive and typically requires a small timestep. Run a sweep and find the largest timestep where the integration stays stable. Think about how you would do this. _Before_ you run the code, predict what you expect to see. Get into the habit of predicting what you expect to happen instead of just reacting to bugs when you see them (there is strong evidence for why this is powerful in the theory of Reinforcement Learning).

Next, look at how the pendulum model is encapsulated in the `models` module, and encapsulate the Euler integrator into a similar module.
You should be able to import it as `from integrators import explicit_euler as integrator`.
Before implementing, consider the inputs and outputs, and how you will use it. This will determine the function's signature, e.g. `my_function(a, b, c)`.

Next, implement a [fourth-order Runge-Kutta scheme](https://en.wikipedia.org/wiki/Runge–Kutta_methods) method with the same signature, so you can drop it into the script as `from integrators import rk4 as integrator`, and not have to change the rest of the script. Find the largest timestep, and then use [`timeit`](https://www.geeksforgeeks.org/python/timeit-python-examples/) to compare how fast the code is with each integrator. Do the comparison once using the same-sized timestep, and once using the largest timestep that stayed numerically accurate for each integrator.

## Implement a new model: bouncing ball

Your final task is to implement the dynamics of a bouncing ball.
Similar to the pendulum, you'll want to run some sanity checks to make sure it works well.
Before implementing, briefly describe the dynamics you expect to see and what checks you think make sense.

## Bonus: polish your tools

I highly recommend changing you shell to zsh instead of bash, and setting up [powerLevel10k](https://github.com/romkatv/powerlevel10k), with the following plugins: `git`, `sh-autosuggestions`, `history-substring-search`, `zsh-syntax-highlighting`.

In VSCode, I recommend the following extensions: `GitLess`, `indent-rainbow`, `Peacock (color-code your editors for different projects)`, `Python Indent`, `Rainbow CSV`, `Trailing Spaces`, `Ruff`.
You can also set up a font that supports ligatures, like [FiraCode](https://github.com/tonsky/FiraCode).

On macOS, get [iTerm2](https://iterm2.com), and [HomeBrew](https://brew.sh).
