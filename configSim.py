import os

HOST = "192.168.112.1" # The remote host
#HOST = "192.168.2.10" # The remote host
PORT = 30011  # The same port as used by the server

MODEL = "models/duplo_efficientdet_lite0.tflite"

# Id of camera
CAMID = 0
# Width of frame to capture from camera
FRAME_WIDTH = 640
# Height of frame to capture from camera
FRAME_HIGHT = 480
# Number of CPU threads to run the model
NUM_THREADS = 4
# Whether to run the model on EdgeTPU
EDGETPU = False
# Wehther to run with UR or without
WITH_UR = False


# Min score for bounding box
MIN_SCORE = 0.8
TARGET_POS = "(0.0, 0.0, 0.0, -3.142, 0.0, 0.0)"
p1 = "(0.1, 0.1, 0.1, 1.142, 0.0, 1.57)"
# reference_point_coordinate_system
RPX = 520
RPY = 75