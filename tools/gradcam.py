import argparse
import cv2
from deepracer_viz.gradcam import load_model_session, gradcam, blend_gradcam_image
import numpy as np


def main(args):
    capture = cv2.VideoCapture(args.input_file)

    CODEC = "X264"
    FPS = args.fps

    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    writer = None

    sess = load_model_session(args.model)

    while capture.isOpened():
        ret, frame = capture.read()

        if ret == True:
            if writer is None:
                writer = cv2.VideoWriter(
                    args.output, fourcc, FPS, (frame.shape[1], frame.shape[0]))

            grad_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cam = gradcam(sess, grad_frame, args.action)
            out_frame = blend_gradcam_image(frame, cam)

            cv2.imshow('frame', out_frame)
            writer.write(out_frame)

            if cv2.waitKey(int(np.ceil(1000.0 / FPS))) & 0xFF == ord('q'):
                break
        else:
            break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_file", help="The name of the MP4 file to process.")
    parser.add_argument(
        "-m", "--model", help="The .pb file containing the model.")
    parser.add_argument(
        "-a", "--action", required=True, type=int, help="The index of the action in the action space to visualize")
    parser.add_argument(
        "--fps", help="FPS of the output video.", type=int, default=15)
    parser.add_argument(
        "-o", "--output", required=True, help="MP4 output file to store the gradcam output.")

    args = parser.parse_args()

    main(args)
