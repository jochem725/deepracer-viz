import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import os


def load_model_session(pbpath: str):
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True))

    with gfile.FastGFile(pbpath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    return sess


def gradcam(sess: tf.Session, input_frame, action_idx):

    input_img = cv2.resize(input_frame, (160, 120))
    input_img = np.expand_dims(input_img, axis=2)

    input_layer = sess.graph.get_tensor_by_name(
        'main_level/agent/main/online/network_0/observation/observation:0')
    output_layer = sess.graph.get_tensor_by_name(
        'main_level/agent/main/online/network_1/ppo_head_0/policy:0')
    convolutional_output = sess.graph.get_tensor_by_name(
        'main_level/agent/main/online/network_1/observation/Conv2d_4/Conv2D:0')

    feed_dict = {input_layer: [input_img]}

    # Get output for this action
    y_c = tf.reduce_sum(tf.multiply(output_layer,  tf.one_hot(
        [action_idx], output_layer.shape[-1])), axis=1)

    # Compute gradients based on last cnn layer
    target_grads = tf.gradients(y_c, convolutional_output)[0]

    out, grads_value = sess.run(
        [convolutional_output, target_grads], feed_dict=feed_dict)
    out, grads_value = out[0, :], grads_value[0, :, :, :]

    weights = np.mean(grads_value, axis=(0, 1))
    cam = np.dot(out, weights)

    # ReLU (only positive values are of interest)
    cam = np.maximum(0, cam)

    # Postprocess
    # Scale maximum value to 1.0
    cam = cam / np.max(cam)

    # Scale back to input frame dimensions.
    input_h, input_w = input_frame.shape[:2]
    cam = cv2.resize(cam, (input_w, input_h))

    return cam


def blend_gradcam_image(image, cam):
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)

    return np.uint8(cam)
