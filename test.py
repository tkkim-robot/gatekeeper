import numpy as np
import matplotlib.pyplot as plt

"""
Created on April 14th, 2025
@author: Taekyung Kim

@description: 
A simple double Integrator model for testing. The code borrowed from "tkkim-robot/safe_control" repository.
"""

class DoubleIntegrator2D:

    def __init__(self, dt, robot_spec):
        '''
            X: [x, y, vx, vy]
            theta: yaw angle
            U: [ax, ay]
        '''
        self.dt = dt
        self.robot_spec = robot_spec

        if 'a_max' not in self.robot_spec:
            self.robot_spec['a_max'] = 1.0
        if 'v_max' not in self.robot_spec:
            self.robot_spec['v_max'] = 1.0
        if 'ax_max' not in self.robot_spec:
            self.robot_spec['ax_max'] = self.robot_spec['a_max']
        if 'ay_max' not in self.robot_spec:
            self.robot_spec['ay_max'] = self.robot_spec['a_max']

    def f(self, X):
        return np.array([X[2, 0],
                            X[3, 0],
                            0,
                            0]).reshape(-1, 1)

    def df_dx(self, X):
        return np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

    def g(self, X):
        return np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

    def step(self, X, U):
        X = X + (self.f(X) + self.g(X) @ U) * self.dt
        return X
    
    def nominal_input(self, X, G, d_min=0.05, k_v=1.0, k_a=1.0):
        '''
        nominal input for CBF-QP (position control)
        '''
        G = np.copy(G.reshape(-1, 1))  # goal state
        v_max = self.robot_spec['v_max']  # Maximum velocity (x+y)
        a_max = self.robot_spec['a_max']  # Maximum acceleration

        pos_errors = G[0:2, 0] - X[0:2, 0]
        pos_errors = np.sign(pos_errors) * \
            np.maximum(np.abs(pos_errors) - d_min, 0.0)

        # Compute desired velocities for x and y
        v_des = k_v * pos_errors
        v_mag = np.linalg.norm(v_des)
        if v_mag > v_max:
            v_des = v_des * v_max / v_mag

        # Compute accelerations
        current_v = X[2:4, 0]
        a = k_a * (v_des - current_v)
        a_mag = np.linalg.norm(a)
        if a_mag > a_max:
            a = a * a_max / a_mag

        return a.reshape(-1, 1)

    def stop(self, X, k_a=1.0):
        # Set desired velocity to zero
        vx_des, vy_des = 0.0, 0.0
        ax = k_a * (vx_des - X[2, 0])
        ay = k_a * (vy_des - X[3, 0])
        return np.array([ax, ay]).reshape(-1, 1)

    def has_stopped(self, X, tol=0.05):
        return np.linalg.norm(X[2:4, 0]) < tol

class BaseRobot:

    def __init__(self, X0, robot_spec, dt, ax):
        self.X = X0.reshape(-1, 1)
        self.dt = dt
        self.robot_spec = robot_spec
        colors = plt.colormaps.get_cmap('Pastel1').colors  # color palette

        color = colors[self.robot_spec['robot_id'] % len(colors) + 1]

        if 'radius' not in self.robot_spec:
            self.robot_spec['radius'] = 0.25
        self.robot_radius = self.robot_spec['radius']  # including padding
        self.robot = DoubleIntegrator2D(dt, robot_spec)

        self.U = np.array([0, 0]).reshape(-1, 1)
        # Plot handles
        self.vis_orient_len = 0.5
        self.body = ax.add_patch(plt.Circle(
            (0, 0), self.robot_radius, edgecolor='black', facecolor=color, fill=True))

        # Robot's orientation axis represented as a line
        self.axis,  = ax.plot([self.X[0, 0], self.X[0, 0]], [
                      self.X[1, 0], self.X[1, 0]], color='r', linewidth=2)

    def get_position(self):
        return self.X[0:2].reshape(-1)
    
    def f(self):
        return self.robot.f(self.X)

    def g(self):
        return self.robot.g(self.X)

    def nominal_input(self, goal, d_min=0.05, k_omega = 2.0, k_a = 1.0, k_v = 1.0):
        return self.robot.nominal_input(self.X, goal, d_min, k_v, k_a)

    def stop(self):
        return self.robot.stop(self.X)

    def has_stopped(self):
        return self.robot.has_stopped(self.X)
    
    def step(self, U):
        # wrap step function
        self.U = U.reshape(-1, 1)
        self.X = self.robot.step(self.X, self.U)
        return self.X

    def render_plot(self):
        self.body.center = self.X[0, 0], self.X[1, 0]

        self.axis.set_ydata([self.X[1, 0], self.X[1, 0]])
        self.axis.set_xdata([self.X[0, 0], self.X[0, 0] ])

# Test simulation using the Gatekeeper controller.
if __name__ == "__main__":
    from gatekeeper import Gatekeeper
    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)

    dt = 0.05
    tf = 100
    num_steps = int(tf / dt)

    # Define an obstacle for visualization.
    obstacle_center = np.array([5.0, 5.0])
    obstacle_radius = 1.0
    nearest_obs = np.array([[5, 5, 1]])

    # Set start and goal positions.
    initial_state = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)
    goal = np.array([10.0, 10.0]).reshape(-1, 1)

    # Create robot_spec and initialize the double integrator.
    robot_spec = {"model": "DoubleIntegrator2D",
                  "robot_id": 1,
                  "radius": 0.25,
                  "v_max": 3.0}
    robot = BaseRobot(initial_state, robot_spec, dt, ax)

    # Instantiate Gatekeeper controller.
    gk = Gatekeeper(robot, dt=dt, nominal_horizon=2.0, backup_horizon=4.0, event_offset=1.0)
    # Set the nominal and backup controllers.
    gk._set_nominal_controller(robot.robot.nominal_input)
    gk._set_backup_controller(robot.robot.stop)
    
    # Control reference dictionary.
    control_ref = {'goal': goal, 'state_machine': 'track', 'u_ref': np.zeros((2, 1))}

    # Simulation loop.
    for step in range(num_steps):
        # Get control input from Gatekeeper.
        u = gk.solve_control_problem(robot.X, control_ref, nearest_obs)
        # Update the robot's state.
        robot.step(u)
        # Render the updated robot state.
        robot.render_plot()
        
        # (Optional) Draw the obstacle once. To avoid redrawing multiple times, we draw it in the first iteration.
        if step == 0:
            obstacle_patch = plt.Circle((obstacle_center[0], obstacle_center[1]), obstacle_radius,
                                          edgecolor='red', facecolor='none', linestyle='--')
            ax.add_patch(obstacle_patch)
        
        plt.pause(0.001)
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    plt.ioff()
    plt.show()