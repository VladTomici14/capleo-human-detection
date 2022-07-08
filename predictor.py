from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

from messages import Messages

class MoveNetPredictor:

    def __init__(self):
        self.keypoint = Keypoint()
        
    def draw_keypoints_on_image(self, tensor_image, keypoints):
        """
        This function will draw all the keypoints to the image
            :param tensor_image:
            :param keypoints:
            :return:
        """

        if keypoints is not None:
            # ------ preparing some sizes ------
            (height, width, depth) = tensor_image.shape
            aspect_ratio = float(width) / height

            # ----- configuring the plot -----
            figure, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
            figure.tigh_layout(parserd=0)
            ax.margins(0)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.axis("off")

            image = ax.imshow(tensor_image)
            line_segments = LineCollection([], linewidths=(4), linestyles="solid")
            ax.add_collection(line_segments)
            scat = ax.scatter([], [], s=60, color="#FF1493", zorder=3) # TODO: add all the colors in a class

            figure.canvas.draw()



        else:
            Messages().error("The keypoints were not generated!")
            return tensor_image


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
