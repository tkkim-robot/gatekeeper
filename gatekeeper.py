"""
Created on April 14th, 2025
@author: Taekyung Kim

@description:
Python implementation of gatekeeper that guarantees safety for all time for safety-critical planning and control.
It manages candidate trajectory generation and updates the committed trajectory.
The controller switches between a nominal trajectory and a backup trajectory according to a pre-defined candidate horizon and event offset.

This version supports two modes:
1. Forward propagation mode: If nominal_controller and backup_controller are provided, they are used to generate candidate trajectories
   (as dictionaries containing trajectory arrays, e.g. {'states': [...], 'controls': [...]}) via an IVP integration.
2. External trajectory mode: If the controllers are MPC-based and do not provide a function for generating trajectories, then an external
   committed trajectory (an array of control inputs) is assumed to be updated outside gatekeeper. In that case, gatekeeper will simply
   return the next control input from the external trajectory.
"""
import numpy as np
from scipy.integrate import solve_ivp


class Gatekeeper:
    def __init__(self, robot, dt=0.05, nominal_horizon=2.0, backup_horizon=2.0, event_offset=1.0):
        """
        Initialize the Gatekeeper controller.

        Parameters:
            robot: An instance providing methods such as nominal_input(state, goal) and stop(state).
                   It must have attributes X (current state) and goal.
            dt: Simulation timestep.
            nominal_horizon: Duration (in seconds) during which the nominal trajectory is applied (TS in paper).
            backup_horizon: Duration (in seconds) for the backup trajectory (TB in paper).
            event_offset: Time offset to push the next candidate event.
            nominal_controller: Optional function that takes (current_state, goal, horizon, dt) and returns a dict with keys
                                      'states' and 'controls' representing the nominal trajectory over the horizon.
            backup_controller: Optional function that takes (current_state, goal, horizon, dt) and returns a dict with keys
                                      'states' and 'controls' representing the backup trajectory over the horizon.
            external_committed_trajectory: If provided (for MPC-based controllers), this is an externally updated list (or array)
                                      of control inputs representing the committed trajectory.

        Mode 1 (forward propagation) is used if nominal_controller and backup_controller are provided.
        Otherwise, mode 2 (external trajectory mode) is assumed.

        Convention:
             time: in seconds and need to devide by dt to get index
             time_index: integer
        """
        self.robot = robot
        self.dt = dt
        self.nominal_horizon = nominal_horizon
        self.backup_horizon = backup_horizon
        self.event_offset = event_offset
        self.horizon_discount = dt * 5


        self.nominal_controller = None
        self.backup_controller = None

        self.next_event_time = 0.0
        self.current_time_idx = backup_horizon / dt # start from backup trajectory. If the initial canddidate traj is valid, it will be updated to 0
        self.committed_horizon = 0.0

        # Initialize the committed trajectory
        self.committed_x_traj = None
        self.committed_u_traj = None

    def _dynamics(self, x, u):
        x_col = np.array(x).reshape(-1, 1)
        # Compute the derivative: f(x) + g(x) @ u.
        dx = self.robot.robot.f(x_col) + self.robot.robot.g(x_col) @ u
        return dx.flatten()

    def _set_nominal_controller(self, nominal_controller):
        self.nominal_controller = nominal_controller

    def _set_backup_controller(self, backup_controller):
        self.backup_controller = backup_controller

    def _set_nominal_trajectory(self, nominal_x_traj, nominal_u_traj):
        """
        Set the nominal trajectory for external trajectory mode.
        ex) using MPC-based controller
        """
        self.nominal_x_traj = nominal_x_traj
        self.nominal_u_traj = nominal_u_traj

    def _generate_nominal_trajectory(self, initial_state, goal, horizon):
        # Create the time evaluation points.
        t_eval = np.linspace(0, horizon, int(horizon / self.dt) + 1)

        # Define an ODE function that incorporates feedback via the nominal controller.
        def f_nom(t, x):
            x_col = np.array(x).reshape(-1, 1)
            u = self.nominal_controller(x_col, goal)
            return self._dynamics(x, u)
        
        #print("initial_state", initial_state.flatten(), initial_state.shape)

        sol = solve_ivp(f_nom, [0, horizon], initial_state.flatten(), t_eval=t_eval, vectorized=False)
        x_traj = sol.y.T
        #print("x_traj", x_traj, x_traj.shape)
        # Now compute the control trajectory by applying the nominal controller to each state.
        u_traj = [self.nominal_controller(np.array(x).reshape(-1,1), goal).squeeze() for x in x_traj]
        u_traj = np.array(u_traj) # (n_traj, n_u)
        return x_traj, u_traj

    def _generate_backup_trajectory(self, initial_state, goal, horizon):
        # don't use goal for backup trajectory in this implementation
        t_eval = np.linspace(0, horizon, int(horizon / self.dt) + 1)

        def f_backup(t, x):
            # Compute the backup control; goal is not used.
            x_col = np.array(x).reshape(-1, 1)
            u = self.backup_controller(x_col)
            return self._dynamics(x, u)

        sol = solve_ivp(f_backup, [0, horizon], initial_state.flatten(), t_eval=t_eval, vectorized=False)
        x_traj = sol.y.T[1:]  # Exclude the state at backup time (duplicate from nominal)
        u_traj = [self.backup_controller(np.array(x).reshape(-1,1)).squeeze() for x in x_traj]
        u_traj = np.array(u_traj)
        return x_traj, u_traj
        
    def _generate_candidate_trajectory(self, goal, discounted_nominal_horizon):
        if self.nominal_controller is None and self.backup_controller is None:
            # if no controllers are provided, return current candidate trajectory
            # in this case, the nominal trajectory is assumed to be externally updated
            nominal_x_traj = self.nominal_x_traj[:discounted_nominal_horizon//self.dt]
            nominal_u_traj = self.nominal_u_traj[:discounted_nominal_horizon//self.dt]
        else:
            # Generate the candidate trajectory using the nominal and backup controllers
            current_state = self.robot.X
            nominal_x_traj, nominal_u_traj = self._generate_nominal_trajectory(current_state, goal, discounted_nominal_horizon)

        state_at_backup = nominal_x_traj[-1]  # last state of the nominal trajectory
        backup_x_traj, backup_u_traj = self._generate_backup_trajectory(state_at_backup, goal, self.backup_horizon)

        self.candidate_x_traj = np.vstack((nominal_x_traj, backup_x_traj))
        self.candidate_u_traj = np.vstack((nominal_u_traj, backup_u_traj))
        # print("nominal_x_traj", nominal_x_traj)
        # print("backup_x_traj", backup_x_traj)
        # print("candidate_x_traj", self.candidate_x_traj)
        return self.candidate_x_traj

    def _is_collision(self, state, obs):
        # obs has x, y, radius, check collision use two norm
        obsX = obs[0:2]
        d_min = obs[2] + self.robot.robot_radius  # obs radius + robot radius
        h = np.linalg.norm(state[0:2] - obsX[0:2])**2 - d_min**2
        return h < 0
    
    def _is_candidate_valid(self, candidate_x_traj, unsafe_region):
        """
        Check if the candidate trajectory is valid by evaluating the safety condition.
        """
        # if unsafe region is None or empty, return True
        if unsafe_region is None or len(unsafe_region) == 0:
            return True

        # Check if the candidate trajectory is within the safe region
        for state in candidate_x_traj:
            for obs in unsafe_region:
                if self._is_collision(state, obs):
                    return False
        return True

    def _update_committed_trajectory(self, discounted_nominal_horizon):
        """
        Update the committed trajectory with the candidate trajectory.
        """
        self.committed_x_traj = self.candidate_x_traj
        self.committed_u_traj = self.candidate_u_traj
        self.next_event_time = self.event_offset
        self.current_time_idx = 0
        self.committed_horizon = discounted_nominal_horizon


    def solve_control_problem(self, robot_state, control_ref, nearest_obs):
        """
        This method should be called at every control timestep.
        It manages event timing: if current time exceeds the event trigger, it generates a new candidate trajectory.
        Then it outputs the control input by selecting from the committed trajectory.

        Returns:
            control_input: The control output for the current timestep.
        """
        self.current_time_idx += 1
        goal = control_ref['goal']

        if control_ref['state_machine'] != 'track':
            # if outer loop is doing something else, just return the reference
            return control_ref['u_ref']
        
        if self.committed_x_traj is None and self.committed_u_traj is None:
            # initialize the committed trajectory
            init_x_traj, init_u_traj = self._generate_backup_trajectory(robot_state, goal, self.backup_horizon)   
            self.committed_x_traj = init_x_traj
            self.committed_u_traj = init_u_traj 

        # try updating the committed trajectory
        if self.current_time_idx > self.next_event_time/self.dt:
            #print("Event triggered, generating new candidate trajectory")

            for i in range(int(self.nominal_horizon//self.horizon_discount)):
                # discount the nominal horizon
                discounted_nominal_horizon = self.nominal_horizon - i * self.horizon_discount
                # Generate the candidate trajectory
                candidate_x_traj = self._generate_candidate_trajectory(goal, discounted_nominal_horizon)
                # Check if the candidate trajectory is valid
                if self._is_candidate_valid(candidate_x_traj, nearest_obs):
                    #print("Candidate trajectory is valid")
                    self._update_committed_trajectory(discounted_nominal_horizon)
                    break
        
        if self.current_time_idx < self.committed_horizon/self.dt:
            # Use the committed trajectory for the next control input
            #print("in nominal: control input", self.committed_u_traj[self.current_time_idx])
            #print("in nominal: control input", self.nominal_controller(self.robot.X, goal))
            return self.committed_u_traj[self.current_time_idx] if self.nominal_controller is None else self.nominal_controller(self.robot.X, goal)
        else:
            #print("backup: control input", self.backup_controller(self.robot.X))
            return self.committed_u_traj[self.current_time_idx] if self.backup_controller is None else self.backup_controller(self.robot.X)