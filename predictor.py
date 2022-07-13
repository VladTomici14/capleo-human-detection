from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from messages import Messages
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2


# TODO: work on the style of the drawings of lines and points
# TODO: calculate the radius of the circles and the thinckness of the lines using the picture's shapes

class MoveNetPredictor:

    def __init__(self):
        self.keypoint = Keypoint()

    def detect_on_image(self, image, interpreter, input_size, drawing=True):
        """
        This function will return the model keypoints for an image
            :param image: the image that we want to do the detection on
            :param interpreter: the interpreter that we want to use for the detection
            :param drawing: the condition if we want to draw on the image that was used as an argument
            :return: the input image but with all the lines and points drawn
        """

        frame = image.copy()
        frame = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), input_size, input_size)
        input_image = tf.cast(frame, dtype=tf.float32)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]["index"])

        self.draw_kpts(image, keypoints_with_scores, self.keypoint.EDGES, confidence_threshold=0.3)

        return image

    def draw_kpts(self, frame, keypoints, edges, confidence_threshold):
        """
        This function will draw the points and lines on the frame
            :param frame: the canvas frame, where we will draw the lines and points
            :param keypoints: the coordinates and the confidence of each point
            :param edges: the color and the coordinated for each line
            :param confidence_threshold: the threshold for knowing if the confidence of a point is right
            :return: the drawn input frame
        """

        (height, width) = frame.shape[:2]
        shaped = np.squeeze(np.multiply(keypoints, [height, width, 1]))

        # ------ drawing the lines -----
        for edge, color in edges.items():
            (p1, p2) = edge
            (y1, x1, c1) = shaped[p1]
            (y2, x2, c2) = shaped[p2]

            if c1 > confidence_threshold and c2 > confidence_threshold:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 6)

        # ------ drawing the points -----
        for keypoint in shaped:
            (kpts_y, kpts_x, kpts_confidence) = keypoint
            if kpts_confidence > confidence_threshold:
                cv2.circle(frame, (int(kpts_x), int(kpts_y)), 8, (0, 0, 255), -1)





class Keypoint:
    KEYPOINT_DICT = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16
    }

    # Maps bones to a matplotlib color name.
    EDGES = {
        (0, 1): 'm',
        (0, 2): 'c',
        (1, 3): 'm',
        (2, 4): 'c',
        (0, 5): 'm',
        (0, 6): 'c',
        (5, 7): 'm',
        (7, 9): 'm',
        (6, 8): 'c',
        (8, 10): 'c',
        (5, 6): 'y',
        (5, 11): 'm',
        (6, 12): 'c',
        (11, 12): 'y',
        (11, 13): 'm',
        (13, 15): 'm',
        (12, 14): 'c',
        (14, 16): 'c'
    }
