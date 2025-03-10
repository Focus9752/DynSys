import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import HTML

# Parameters
s=10
r=28
b=2.667

# Fixed points
fp_arr = [[0, 0, 0], [np.sqrt(b*(r-1)), np.sqrt(b*(r-1)), r-1], [-np.sqrt(b*(r-1)), -np.sqrt(b*(r-1)), r-1]]

# Initial state
r_vec_initial = [0, 1, 10]

# Time step size
dt = 0.01

# Stop time
t_stop = 100

# Time array
t_arr = np.arange(0, t_stop, dt)
n_time_steps = len(t_arr)

# Colours
BACKGROUND_COLOR = "black"
PLOT_COLOR = "grey"

# Width of the plotted trajectory
LW = 1

# Camera rotation speed
ROTATION_SPEED = 1/10

# Integrate the Lorenz equations with the RK4 method
def lorenz(r_vec):
    x, y, z = r_vec
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y -b*z

    r_dot_vec = np.array([x_dot, y_dot, z_dot])

    return r_dot_vec

# RK4 implementation
def RK4(r_vec):
    k1 = lorenz(r_vec) * dt
    k2 = lorenz(r_vec + 1/2 * k1) * dt
    k3 = lorenz(r_vec + 1/2 *k2) *dt
    k4 = lorenz(r_vec + k3) * dt 

    return r_vec + 1/6*(k1 + 2*k2 + 2*k3 + k4)

# There are n_time_step states of the system, each with three coordiantes, so we store the states in a n x 3 array
pos = np.zeros((n_time_steps, 3))

# Apply initial conditions
for i in range(3):
    pos[0, i] = r_vec_initial[i]

# The first element is determined by the initial conditions, so we update the i + 1st element
for i in range(n_time_steps - 1):
    pos[i + 1] = RK4(pos[i])

fig = plt.figure(facecolor=BACKGROUND_COLOR, figsize=(10,10))
ax = fig.add_subplot(111, projection="3d")

# Update function for the animation
def update(i):
    # Sets the viewpoint of the camera
    ax.view_init(0, -56 + i*ROTATION_SPEED) # + i*ROTATION_SPEED to rotate around the z-axis
    ax.clear()
    # Plot fixed points
    ax.scatter(fp_arr[0][0], fp_arr[0][1], fp_arr[0][2], color="red", label="Fixed Point #1")
    ax.scatter(fp_arr[1][0], fp_arr[1][1], fp_arr[1][2], color="lime", label="Fixed Point #2")
    ax.scatter(fp_arr[2][0], fp_arr[2][1], fp_arr[2][2], color="blue", label="Fixed Point #3")
    
    ax.set(facecolor=BACKGROUND_COLOR)
    ax.set_axis_off() # hides axes
    ax.plot(pos[:i, 0], pos[:i, 1], pos[:i, 2], color=PLOT_COLOR, linewidth=LW, label="Trajectory") # plot the trajectory up to the current point
    ax.scatter(pos[i, 0], pos[i, 1], pos[i, 2], color="white", linewidth=LW, marker="*")
    ax.legend()

ani = animation.FuncAnimation(fig, update, frames=n_time_steps, interval = 10, repeat = False)
plt.show()

