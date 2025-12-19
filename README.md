# gatekeeper

This repository is archived now. Please see [safe_control](https://github.com/tkkim-robot/safe_control) for more details.

### Installation
Install packages
```bash
python -m pip install numpy scipy matplotlib
```

### Run basic test
```python
python test.py
```

The sample results from the basic example:

|      Navigation with gatekeeper            |
| :-------------------------------: |
|  <img src="https://github.com/user-attachments/assets/797fa7c2-8e8e-46d0-81ab-a4b47fdd8ec5"  height="350px"> |

### How to use it

```python
from gatekeeper import Gatekeeper

class TrackingController:
    def __init__(self, robot):
        self.robot = robot
        # Instead of using a pre-existing positional controller,
        # instantiate the gatekeeper controller.
        self.gk = Gatekeeper(robot, dt=0.05, nominal_horizon=2.0, backup_horizon=4.0, event_offset=1.0)

        # define your nominal and backup controller
        self.gk._set_nominal_controller(robot.robot.nominal_input)
        self.gk._set_backup_controller(robot.robot.stop)


    def solve_control_problem(self, current_state, current_time):
        # The gatekeeper's solve_control_problem is called at each time step.
        control_input = self.tracking_controller.solve_control_problem(current_state, current_time)
        return control_input

```
