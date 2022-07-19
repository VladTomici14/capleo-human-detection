import tensorflow as tf
import numpy as np
import cv2
from utils.angles import Geometry
from utils.colors import BGRColors

# TODO: work on the style of the drawings of lines and points
# TODO: calculate the radius of the circles and the thinckness of the lines using the picture's shapes
# TODO: add confidence threshold to any thing

class MoveNetPredictor:

    def __init__(self, model_path, input_size):
        self.shaped = None
        self.keypoint = Keypoint()
        self.input_size = input_size

        # ------ loading the model ------
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def detect_on_image(self, image, drawing=True, angles=True):
        """
        This function will return the model keypoints for an image
            :param image: the image that we want to do the detection on
            :param drawing: the condition if we want to draw on the image that was used as an argument
            :param angles: the condition if we want to show the angles values
            :return: the input image but with all the lines and points drawn
        """

        # ------ preparing the image ------
        frame = cv2.resize(image, (self.input_size, self.input_size))
        frame = tf.expand_dims(frame, axis=0)
        input_image = tf.cast(frame, dtype=tf.float32)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # FIXME: debug this weird error (when changing the model to the 4th version)
        self.interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        self.interpreter.invoke()
        keypoints_with_scores = self.interpreter.get_tensor(output_details[0]["index"])

        # ---- drawing the keypoints on the image -----
        if drawing is True:
            self.draw_kpts(image, keypoints_with_scores, self.keypoint.EDGES, 0.3, angles)

        return image

    def draw_kpts(self, frame, keypoints, edges, confidence_threshold, calculate_angles):
        """
        This function will draw the points and lines on the frame
            :param frame: the canvas frame, where we will draw the lines and points
            :param keypoints: the coordinates and the confidence of each point
            :param edges: the color and the coordinated for each line
            :param confidence_threshold: the threshold for knowing if the confidence of a point is right
            :param calculate_angles: the condition if we want to show the angles values
            :return: the drawn input frame
        """

        (self.height, self.width) = frame.shape[:2]
        self.shaped = np.squeeze(np.multiply(keypoints, [self.height, self.width, 1]))

        # ------ drawing the lines -----
        for edge, color in edges.items():
            (p1, p2) = edge
            (y1, x1, c1) = self.shaped[p1]
            (y2, x2, c2) = self.shaped[p2]

            if c1 > confidence_threshold and c2 > confidence_threshold:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 6)

        # ------ drawing the points -----
        k = 0
        for keypoint in self.shaped:
            (kpts_y, kpts_x, kpts_confidence) = keypoint
            if kpts_confidence > confidence_threshold:

                if calculate_angles is True:
                    if k == 9:
                        # ------- printing the angle values between the points that we want to use -----
                        for points in self.keypoint.ANGLES:
                            self.draw_angle_kpts(frame, points)
                            angle = 180 - Geometry(self.shaped).calculate_angle([point for point in points])
                            print(f"{points}: {angle}")
                            if 135 <= angle <= 180:
                                (y1, x1) = self.shaped[points[0]][:2]
                                (y2, x2) = self.shaped[points[2]][:2]
                                (yc, xc) = self.shaped[points[1]][:2]

                                cv2.line(frame, (int(x1), int(y1)), (int(xc), int(yc)), BGRColors().GREEN, 8)
                                cv2.line(frame, (int(x2), int(y2)), (int(xc), int(yc)), BGRColors().GREEN, 8)

                else:
                    head_2_body_angle = Geometry(self.shaped).head_to_body_angle()
                    print(f"head 2 body angle: {head_2_body_angle}")
                    cv2.circle(frame, (int(kpts_x), int(kpts_y)), 8, BGRColors().RED, -1)

                k += 1

    def draw_angle_kpts(self, image, points):
        for index in points:
            (y, x) = self.shaped[index][:2]
            cv2.circle(image, (int(x), int(y)), 8, BGRColors().YELLOW, -1)


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
        (0, 1): BGRColors().PINK,
        (0, 2): BGRColors().PINK,
        (1, 3): BGRColors().PINK,
        (2, 4): BGRColors().PINK,
        (0, 5): BGRColors().GREEN,
        (0, 6): BGRColors().GREEN,
        (5, 7): BGRColors().ORANGE,
        (7, 9): BGRColors().ORANGE,
        (6, 8): BGRColors().YELLOW,
        (8, 10): BGRColors().YELLOW,
        (5, 6): BGRColors().GREEN,
        (5, 11): BGRColors().GREEN,
        (6, 12): BGRColors().GREEN,
        (11, 12): BGRColors().GREEN,
        (11, 13): BGRColors().CYAN,
        (13, 15): BGRColors().CYAN,
        (12, 14): BGRColors().PURPLE,
        (14, 16): BGRColors().PURPLE
    }

    ANGLES = [
        [6, 8, 10],    # the right elbow angle joints
        [5, 7, 9],     # the left elbow angle joints
        [12, 14, 16],  # the right knee angle joints
        [11, 13, 15]   # the left knee angle joints
    ]
