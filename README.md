# gatekeeper

### Installation
Install packages
```bash
python -m pip install numpy scipy matplotlib
```

```python
from gatekeeper import Gatekeeper
from robots.robot import BaseRobot  # your base robot class

# Example use within the tracking module
class TrackingController:
    def __init__(self, robot):
        self.robot = robot
        # Instead of using a pre-existing positional controller,
        # instantiate the gatekeeper controller.
        self.tracking_controller = Gatekeeper(robot, dt=0.05, candidate_horizon=2.0, event_offset=0.05)

    def solve_control_problem(self, current_state, current_time):
        # The gatekeeper's solve_control_problem is called at each time step.
        control_input = self.tracking_controller.solve_control_problem(current_state, current_time)
        return control_input

```