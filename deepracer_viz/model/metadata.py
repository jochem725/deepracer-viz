import json
from .actionspace import ContinuousActionSpace, DiscreteActionSpace

class ModelMetadata:
    def __init__(self, sensor, network, simapp_version, action_space_type, action_space):
        self.sensor = sensor
        self.network = network
        self.simapp_version = simapp_version

        self.action_space_type = action_space_type

        if self.action_space_type == 'continuous':
            self.action_space = ContinuousActionSpace(action_space)
        elif self.action_space_type == 'discrete':
            self.action_space = DiscreteActionSpace(action_space)
        else:
            raise Exception("Unsupported action space type.")

    def __str__(self):
        return "{} -- {} -- SIMAPP_VERSION {}".format(
            self.sensor, self.network, self.simapp_version
        )

    def input_type(self):
        # Currently only support old observation or single camera
        # TODO: Check how we can do this more smart and support stereo.
        input_type = None
        if "observation" in self.sensor:
            input_type = "observation"
        elif "FRONT_FACING_CAMERA" in self.sensor:
            input_type = "FRONT_FACING_CAMERA"
        else:
            raise Exception("Metadata contains unsupported sensor.")

        return input_type

    @staticmethod
    def from_file(model_metadata_path: str):
        """Load a model metadata file
        Args:
            model_metadata_path (str): Path to the model_metadata.json file.
        Raises:
            Exception: If metadata cannot be loaded from the file.
        Returns:
            [tuple]: model sensors, network type, simapp version.
        """

        try:
            with open(model_metadata_path, "r") as json_file:
                data = json.load(json_file)
                if "version" in data:
                    simapp_version = data["version"]
                else:
                    simapp_version = None

                if "sensor" in data:
                    sensor = data["sensor"]
                else:
                    sensor = ["observation"]
                    simapp_version = "1.0"

                if "neural_network" in data:
                    network = data["neural_network"]
                else:
                    network = "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW"

                if "action_space_type" in data:
                    action_space_type = data["action_space_type"]
                else:
                    action_space_type = 'discrete'

                if "action_space" in data:
                    action_space = data["action_space"]
                else:
                    raise Exception("No action space in file")

            return ModelMetadata(sensor, network, simapp_version, action_space_type, action_space)
        except Exception as e:
            raise Exception("Error parsing model metadata: {}".format(e))
