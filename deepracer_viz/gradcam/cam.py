import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.python.platform import gfile
import numpy as np
import cv2

from deepracer_viz.model.model import Model

class GradCam:
    def __init__(self, model: Model, target_layers):
        self.model = model

        # Extract gradcam logic from model.
        self.input_layer = self.model.get_model_input()
        self.output_layer = self.model.get_model_output()
        self.target_layers = target_layers
        # self.conv_outputs = self.model.get_model_convolutional_output()

        # Get output for this action
        y_c = tf.reduce_sum(
            tf.multiply(
                self.output_layer,
                tf.one_hot(
                    [0], self.output_layer.shape[-1]
                    # [tf.argmax(self.output_layer)], self.output_layer.shape[-1] [0 for steer, 1 for speed]
                ),  # TODO: Argmax selects target action for PPO, also allow manual action idx to be specified.
            ),
            axis=1,
        )

        # Compute gradients based on last cnn layer
        self.target_grads = tf.gradients(y_c, self.target_layers)

    def process(self, input):
        input_resized = cv2.resize(input, self.model.input_size())
        input_preprocessed = cv2.cvtColor(input_resized, cv2.COLOR_RGB2GRAY)

        input_frame = np.expand_dims(input_preprocessed, axis=2)

        feed_dict = {self.input_layer: [input_frame]}
        result = self.model.session.run(self.output_layer, feed_dict=feed_dict)[0, :]
    
        cams = []
        for target_layer, target_grad in zip(self.target_layers, self.target_grads):
            ops = [target_layer, target_grad]

            out, grads_value = self.model.session.run(ops, feed_dict=feed_dict)
            out, grads_value = out[0, :], grads_value[0, :, :, :]

        
            weights = np.mean(grads_value, axis=(0, 1))
            cam = np.dot(out, weights)

            # ReLU (only positive values are of interest)
            cam = np.maximum(0, cam)

            cam = cam - np.min(cam)
            cam = cam / np.max(1e-7 + cam)

            # Scale back to resized input frame dimensions.
            input_h, input_w = input_resized.shape[:2]
            cam = cv2.resize(cam, (input_w, input_h))
            cam = np.float32(cam)
            
            cams.append(cam[:, None, :])

        cams = np.concatenate(cams, axis=1)
        cams = np.maximum(cams, 0)
        cam = np.mean(cams, axis=1)


        cam = cam - np.min(cam)
        cam = cam / np.max(1e-7 + cam)
        cam = np.float32(cam)

        # Blend
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam = np.float32(cam) + np.float32(input_resized)
        cam = 255 * cam / np.max(cam)
        cam = np.uint8(cam)

        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

        # Scale back to original frame dimensions.
        input_h, input_w = input.shape[:2]
        cam = cv2.resize(cam, (input_w, input_h))

        return result, cam