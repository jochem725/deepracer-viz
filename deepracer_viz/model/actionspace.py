from abc import ABC, abstractmethod
import numpy as np

class ActionSpace(ABC):
    """
    Base class for an action space.
    An action space must have a method to map model output scores to the action values (speed, steering angle)
    """

    @abstractmethod
    def select_action(self, model_output):
        pass

class ContinuousActionSpace(ActionSpace):
    """
    Class for a continuous action space.
    Applies scaling to the output so we get a value within range of the servos. (-1, 1)
    """

    def __init__(self, action_space):
        self.min_steer = action_space['steering_angle']['low']
        self.max_steer = action_space['steering_angle']['high']
        self.min_speed = action_space['speed']['low']
        self.max_speed = action_space['speed']['high']

    def _scale_continuous_value(self, action, min_old, max_old, min_new, max_new): 
        # Model outputs high and low values out of the range; need to clip the action.
        action = np.clip(action, -1.0, 1.0)

        if max_old == min_old:
            print(f"Unsupported minimum and maximum action space bounds for scaling values. min_old: {min_old}; max_old: {max_old}")

        return ((max_new - min_new) / (max_old - min_old)) * (action - min_old) + min_new
        
    def select_action(self, model_output):
        steering_angle = self._scale_continuous_value(model_output[0], -1.0, 1.0, self.min_steer, self.max_steer)
        throttle = self._scale_continuous_value(model_output[1], -1.0, 1.0, self.min_speed, self.max_speed)

        return steering_angle, throttle


class DiscreteActionSpace(ActionSpace):
    """
    Class for a continuous action space.
    Selects the most probably actin from the model output.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, model_output):
        selected_action = self.action_space[np.argmax(model_output)]


        return selected_action['steering_angle'], selected_action['speed']