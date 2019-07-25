import argparse
import cv2
from deepracer_viz.gradcam import load_model_session, gradcam, blend_gradcam_image
import numpy as np
import requests
from bs4 import BeautifulSoup
import time


# Function for processing image and applying GradCAM
# Using GPU helps, make sure to have CUDA installed correctly:
#   You should need CUDA 10: check with `nvcc --version` : https://www.tensorflow.org/install/source#linux
#   https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=debnetwork
#   `sudo apt-get install cuda-libraries-10.0`
def process_image(i, tf_session):
    # Prep image for GradCam
    grad_frame = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    # Get GradCAM
    cam = gradcam(tf_session, grad_frame, args.action)
    # Overlay GradCAM with original image
    return blend_gradcam_image(i, cam)


def main(args):
    global bytes
    CODEC = "X264"
    FPS = args.fps

    # if the ip argument exists then enter into live view mode with video from physical DeepRacer
    if args.ip:
        with requests.Session() as s:
            URL = "https://" + str(args.ip) + "/"
            post_login_url = URL + "/login"
            video_url = URL + "/route?topic=/video_mjpeg&width=480&height=360"

            # Get the CSRF Token
            response = s.get(URL, verify=False)
            soup = BeautifulSoup(response.text, 'lxml')
            csrf_token = soup.select_one('meta[name="csrf-token"]')['content']
            headers = {'X-CSRFToken': csrf_token}

            # Login to the DeepRacer web interface with Post
            if not args.password:
                print("ERROR: User must add password for DeepRacer before stream can be accessed")
                exit(1)
            payload = {'password': args.password}
            post = s.post(post_login_url, data=payload, headers=headers, verify=False)

            # Get the video stream
            video_stream = s.get(video_url, stream=True, verify=False)

            # Load the CNN
            sess = load_model_session(args.model)

            if video_stream.status_code == 200:
                print("Video Connected!")
                last_image_time = time.time()
                # Bytes to build Jpeg
                bytes = bytes()
                for chunk in video_stream.iter_content(chunk_size=1024):
                    bytes += chunk
                    a = bytes.find(b'\xff\xd8')  # Marker byte pair
                    b = bytes.find(b'\xff\xd9')  # Trailing byte pair
                    #  If both byte pairs on in the stream then build the jpeg
                    if a != -1 and b != -1:
                        jpg = bytes[a:b + 2]
                        bytes = bytes[b + 2:]
                        if time.time() - last_image_time > 1.0 / args.fps:
                            i = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            out_frame = process_image(i, sess)
                            # cv2.imshow('Raw Image', i)
                            cv2.imshow('GradCAM', cv2.resize(out_frame, (1920, 1440)))
                            last_image_time = time.time()
                        if cv2.waitKey(1) == 27:  # Press esc to stop processing images
                            break
            else:
                print("Received unexpected status code {}".format(video_stream.status_code))

    else:
        capture = cv2.VideoCapture(args.input_file)
        fourcc = cv2.VideoWriter_fourcc(*CODEC)

        # If an output file is specified then init writer
        if args.output:
            writer = None
        else:
            writer = 1  # This will prevent writer init

        sess = load_model_session(args.model)
        while capture.isOpened():
            ret, frame = capture.read()
            if ret:
                if writer is None:
                    writer = cv2.VideoWriter(
                        args.output, fourcc, FPS, (frame.shape[1], frame.shape[0]))

                # Apply GradCAM
                out_frame = process_image(frame, sess)

                cv2.imshow('frame', out_frame)
                writer.write(out_frame)

                if cv2.waitKey(int(np.ceil(1000.0 / FPS))) & 0xFF == ord('q'):
                    break
            else:
                break

        capture.release()
        # release writer if it was used
        if writer is None:
            writer.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # non-optional arguments
    parser.add_argument("model", help="The .pb file containing the model.")
    parser.add_argument("action", type=int, help="The index of the action in the action space to visualize")

    # args for static input and output video (optional)
    parser.add_argument("-f", "--input_file", help="The name of the MP4 file to process.")
    parser.add_argument("--fps", help="FPS of the output video.", type=int, default=10)
    parser.add_argument("-o", "--output", help="MP4 output file to store the gradcam output.")

    # args for live stream from physical DeepRacer (optional)
    parser.add_argument("-i", "--ip", help="The IP of the DeepRacer if you are using live feed.")
    parser.add_argument("-p", "--password", help="The password for your DeepRacer if doing live feed.")

    args = parser.parse_args()

    main(args)
