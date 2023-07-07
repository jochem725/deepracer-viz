import argparse
import cv2
from deepracer_viz.gradcam.cam import GradCam
from deepracer_viz.model.metadata import ModelMetadata
from deepracer_viz.model.model import Model


import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def gradcam_cmd(input_file, output_file, model, model_metadata, fps=15):
    capture = cv2.VideoCapture(input_file)

    CODEC = "MP4V"
    FPS = fps

    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    writer = None

    metadata = ModelMetadata.from_file(model_metadata)
    model = Model.from_file(model, metadata)

    with model.session as sess:
        cam = GradCam(model, model.get_conv_outputs())
        
        index = 0
        vidsize = (1280, 480)
        while capture.isOpened():
            ret, input_frame = capture.read()

            if ret == True:
                if writer is None:
                    writer = cv2.VideoWriter(
                        output_file, fourcc, FPS, vidsize)

                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                result, out_frame = cam.process(input_frame)
                steer, speed = model.get_action(result)

                # PILLOW CODE
                frame = Image.new('RGBA', vidsize, color='#FFFFFF')
                frame.paste(Image.fromarray(out_frame).resize((640, 480)), box=(0, 0))
                frame.paste(Image.fromarray(input_frame).resize((640, 480)), box=(640, 0))

                draw = ImageDraw.Draw(frame)
                fnt = ImageFont.truetype('/System/Library/Fonts/SFNS.ttf', 18)

                draw.text((10,10), "GradCam", font=fnt, fill=(255,255,255,255))
                draw.text((650,10), "Selected Action", font=fnt, fill=(255,255,255,255))
                draw.text((1180,10), "Frame: {}".format(index), font=fnt, fill=(255,255,255,255))

                draw.text((650,430), "{:.2f} m/s".format(speed), font=fnt, fill=(255,255,255,255))
                draw.text((650,450), "{:.2f} degrees".format(steer), font=fnt, fill=(255,255,255,255))

                frame = np.uint8(np.array(frame.convert("RGB")))
                frame[:,:,[0, 2]] = frame[:,:,[2, 0]]

                # cv2.ellipse(frame, (710,450), (50, 50), 180, 0, 180, [0, 0, 0], 20, cv2.LINE_AA)
                # cv2.ellipse(frame, (710,450), (50, 50), 180, 0, speed / 4.0 * 180, [0, 255, 0], 10, cv2.LINE_AA)

                # cv2.ellipse(frame, (850,450), (50, 50), 180, 0, 180, [0, 0, 0], 20, cv2.LINE_AA)
                # cv2.ellipse(frame, (850,450), (50, 50), 180, round((0.5 + -steer / 60.0) * 180.0, 2),round((0.5 + -steer / 60.0) * 180.0 + 1.0, 2), [0, 255, 0], 10, cv2.LINE_AA)

                # cv2.imshow('frame', frame)
                
                writer.write(frame)
                
                index += 1

                # if cv2.waitKey(int(np.ceil(1000.0 / FPS))) & 0xFF == ord('q'):
                if 0xFF == ord('q'):
                    break
            else:
                break

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

