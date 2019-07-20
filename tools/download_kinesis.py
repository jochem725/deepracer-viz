import argparse
import cv2
from deepracer_viz.kinesis import KinesisVideoStream
import numpy as np


def main(args):
    kinesisStream = KinesisVideoStream(args.stream_name)
    capture = cv2.VideoCapture(kinesisStream.get_live_streaming_session_url())

    CODEC = "H264"
    FPS = args.fps

    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    writer = None

    print("-- Downloading Kinesis stream. Press q to terminate.")

    while capture.isOpened():
        # Capture frame-by-frame
        ret, frame = capture.read()

        if ret == True:
            if writer is None:
                writer = cv2.VideoWriter(
                    args.output, fourcc, FPS, (frame.shape[1], frame.shape[0]))

            cv2.imshow('frame', frame)
            writer.write(frame)

            if cv2.waitKey(int(np.ceil(1000.0 / FPS))) & 0xFF == ord('q'):
                break
        else:
            break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    """
    Tool which can read a live stream from kinesis 
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "stream_name", help="The name of the stream in AWS Kinesis Video.")
    parser.add_argument(
        "-o", "--output", required=True, help="MP4 output file to store the stream.")
    parser.add_argument(
        "--fps", help="FPS of the output video.", type=int, default=15)

    args = parser.parse_args()

    main(args)
