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
            U_attitude: [yaw_rate]
            cbf: h(x) = ||x-x_obs||^2 - beta*d_min^2
            relative degree: 2
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
    


# Test simulation using the Gatekeeper controller.
if __name__ == "__main__":
    plt.ion()
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)

    dt = 0.02
    tf = 20
    num_steps = int(tf / dt)

    # Define an obstacle (for visualization only).
    obstacle_center = np.array([5.0, 5.0])
    obstacle_radius = 1.0

    # Set start and goal positions.
    initial_state = [0.0, 0.0, 0.0, 0.0]  # Start at rest.
    goal = [10.0, 10.0]

    # Create robot_spec and initialize the double integrator.
    robot_spec = {"model": "DoubleIntegrator2D"}
    robot = DoubleIntegrator2D(dt, robot_spec)

    # Import and instantiate the Gatekeeper controller.
    from gatekeeper import Gatekeeper
    gatekeeper_controller = Gatekeeper(robot, dt=dt, candidate_horizon=2.0, event_offset=0.05)

    # Create and plot the robot body (a circle) and an orientation indicator.
    body = ax.add_patch(plt.Circle((robot.X[0, 0], robot.X[1, 0]), robot.robot_radius,
                                   edgecolor='black', facecolor='blue', fill=True))
    orient_line, = ax.plot([robot.X[0, 0], robot.X[0, 0] + robot.vis_orient_len * np.cos(robot.yaw)],
                             [robot.X[1, 0], robot.X[1, 0] + robot.vis_orient_len * np.sin(robot.yaw)], 'r-')

    # Simulation loop.
    for step in range(num_steps):
        current_time = step * dt
        current_state = robot.X

        # Get control input from the Gatekeeper controller.
        control_input = gatekeeper_controller.solve_control_problem(current_state, current_time)

        # Override to stop if near the goal.
        if np.linalg.norm(robot.X[0:2, 0] - np.array(goal)) < 0.5:
            control_input = robot.stop(current_state)
        
        # Update the robot state.
        robot.update_state(control_input)

        # Update visualization.
        body.center = (robot.X[0, 0], robot.X[1, 0])
        orient_line.set_xdata([robot.X[0, 0],
                               robot.X[0, 0] + robot.vis_orient_len * np.cos(robot.yaw)])
        orient_line.set_ydata([robot.X[1, 0],
                               robot.X[1, 0] + robot.vis_orient_len * np.sin(robot.yaw)])

        # Plot the obstacle.
        # (For simplicity we add the obstacle patch at each iteration;
        # in a more efficient implementation, add it once.)
        obstacle_circle = plt.Circle((obstacle_center[0], obstacle_center[1]), obstacle_radius,
                                     edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(obstacle_circle)

        plt.pause(0.001)
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()

    plt.ioff()
    plt.show()