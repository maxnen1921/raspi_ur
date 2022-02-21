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
"""Utility functions to display the pose detection results."""

from typing import List, Tuple

import cv2
import numpy as np
from numpy import ndarray
import configSim

from object_detector import Detection

_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_TEXT_COLOR = (0, 0, 255)  # red

rpx = configSim.RPX
rpy = configSim.RPY
cx = None
cy = None


def visualize(
    image: np.ndarray,
    detections: List[Detection],
    cx=cx,
    cy=cy,
) -> Tuple[ndarray, int, int]:
  """Draws bounding boxes on the input image and return it.

  Args:
    image: The input RGB image.
    detections: The list of all "Detection" entities to be visualize.

  Returns:
    Image with bounding boxes and cx, cy (center of bounding box)
    :param cy:
    :param cx:
  """

  # draw coordinates origin
  cv2.circle(image, (rpx, rpy), 5, (0, 0, 255), -1)
  cv2.circle(image, (rpx, rpy), 15, (0, 0, 255), 2)

  for detection in detections:
    # Draw bounding_box
    start_point = detection.bounding_box.left, detection.bounding_box.top
    end_point = detection.bounding_box.right, detection.bounding_box.bottom
    cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

    # calculate center of bounding box
    cx = int(start_point[0] + (end_point[0] - start_point[0]) / 2)
    cy = int(start_point[1] + (end_point[1] - start_point[1]) / 2)

    # Draw label and score
    category = detection.categories[0]
    class_name = category.label
    probability = round(category.score, 2)
    result_text = class_name + ' (' + str(probability) + ')'
    text_location = (_MARGIN + detection.bounding_box.left, _MARGIN + _ROW_SIZE + detection.bounding_box.top)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, _FONT_SIZE, _TEXT_COLOR, _FONT_THICKNESS)
    cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
    cv2.line(image, (cx, cy), (rpx, rpy), (255, 255, 0), 2)

  return image, cx, cy
