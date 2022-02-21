# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time

import cv2
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
import utils
import configSim
import ur_with_config

import socket
import datetime

model = configSim.MODEL
width = configSim.FRAME_WIDTH
height = configSim.FRAME_HIGHT
num_threads = configSim.NUM_THREADS


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool, ur: bool) -> None:
    """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
    ur: wheter with UR or without
  """

    # start socket connection if running with UR
    if ur:
        print("Starting Socket")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((configSim.HOST, configSim.PORT))
        s.listen(5)
        conn, addr = s.accept()
        time.sleep(0.5)
        print("Socket started")

        robot = ur_with_config.UR()
        targetPos = configSim.TARGET_POS

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model
    options = ObjectDetectorOptions(
        num_threads=num_threads,
        score_threshold=0.3,
        max_results=3,
        enable_edgetpu=enable_edgetpu)
    detector = ObjectDetector(model_path=model, options=options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1
        image = cv2.flip(image, 1)

        # Run object detection estimation using the model.
        detections = detector.detect(image)

        # Draw keypoints and edges on input image
        #if detections:
        image, cx, cy = utils.visualize(image, detections)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

        if ur and detections:
            x_coor, y_coor, z_rot = robot.transform_coordinates_from_pixel(cx, cy)
            new_targetPos = f"({x_coor}, {y_coor}, 0.0, -3.142, 0.0, {z_rot})"
            if counter % 5 == 0 and new_targetPos != targetPos:
                conn.sendall(new_targetPos.encode('utf8'))
                ct = datetime.datetime.now()
                print(f"new_targetPos:{new_targetPos} | ct: {ct}")
                targetPos = new_targetPos

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        cv2.imshow('object_detector', image)

    if ur:
        conn.close()
        s.close()

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, type=int, default=configSim.CAMID)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=configSim.EDGETPU)
    parser.add_argument(
        '--ur',
        help='Whether to run with UR.',
        action='store_true',
        required=False,
        default=configSim.WITH_UR)
    args = parser.parse_args()

    run(model=model, camera_id=int(args.cameraId), width=width, height=height,
        num_threads=num_threads, enable_edgetpu=bool(args.enableEdgeTPU), ur=bool(args.ur))


if __name__ == '__main__':
    main()
