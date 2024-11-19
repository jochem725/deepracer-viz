import tensorflow.compat.v1 as tf1
import re

from deepracer_viz.model.metadata import ModelMetadata

class Model:
    def __init__(self, session, metadata: ModelMetadata):
        self.metadata = metadata
        self.session = session

    def input_size(self):
        input = self.get_model_input()

        height = input.shape[1]
        width = input.shape[2]

        return (width, height)

    def get_conv_outputs(self):
        ops = self.session.graph.get_operations()
        conv_ops = list(filter(lambda x: re.match(r".*Conv2d\_\d+\/BiasAdd", x.name), ops))

        return [op.outputs[0] for op in conv_ops]

    def get_model_input(self):
        ops = self.session.graph.get_operations()

        # Select first operation output tensor.
        return ops[0].outputs[0]

    def get_model_output(self):
        ops = self.session.graph.get_operations()

        # Select last operation output tensor.
        return ops[-1].outputs[0]

    def get_model_convolutional_output(self):
        # Get last convolutional operator.
        print(self.get_conv_outputs())
        return self.get_conv_outputs()[-1]

    def get_action(self, model_output):
        return self.metadata.action_space.select_action(model_output)

    @classmethod
    def from_file(cls, model_pb_path: str, metadata: ModelMetadata, log_device_placement: bool = False):
        """Load the TensorFlow graph for a model.pb model file.
        Args:
            pbpath (str): Path to the model.pb file
        Raises:
            Exception: If the session cannot be loaded from the model file.
        Returns:
            [tf1.Session]: TensorFlow session object.
        """
        try:
            tf1.reset_default_graph()
            sess = tf1.Session(
                config=tf1.compat.v1.ConfigProto(
                    allow_soft_placement=True, log_device_placement=log_device_placement
                )
            )

            with tf1.io.gfile.GFile(model_pb_path, "rb") as f:
                graph_def = tf1.GraphDef()
                graph_def.ParseFromString(f.read())

            sess.graph.as_default()
            tf1.import_graph_def(graph_def, name="")

            return cls(sess, metadata)
        except Exception as e:
            raise Exception("Could not get session for model: {}".format(e))