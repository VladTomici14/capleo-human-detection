from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from messages import Messages
from PIL import Image
import numpy as np
import cv2


class MoveNetPredictor:

    def __init__(self):
        self.keypoint = Keypoint()

    def keypoints_edges_display(self, keypoints, height, width, keypoint_threshold=0.11):
        keypoints_all = []
        edges_all = []
        edge_colors = []
        num_instances = keypoints.shape[0]

        for index in range(num_instances):
            kpts_x = keypoints[0, index, :, 1]
            kpts_y = keypoints[0, index, :, 0]
            kpts_scores = keypoints[0, index, :, 2]

            kpts_absolute_xy = np.stack([width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
            kpts_absolute_above_thresh = kpts_absolute_xy[kpts_scores > keypoint_threshold, :]

            keypoints_all.append(kpts_absolute_above_thresh)

            for edge_pair, color in self.keypoint.KEYPOINT_EDGE_INDS_TO_COLOR.items():
                if (kpts_scores[edge_pair[0]] > keypoint_threshold) and (
                        kpts_scores[edge_pair[1]] > keypoint_threshold):
                    x1 = kpts_absolute_xy[edge_pair[0], 0]
                    y1 = kpts_absolute_xy[edge_pair[0], 1]

                    x2 = kpts_absolute_xy[edge_pair[1], 0]
                    y2 = kpts_absolute_xy[edge_pair[1], 1]

                    line_segment = np.array([[x1, y1], [x2, y2]])

                    edges_all.append(line_segment)
                    edge_colors.append(color)

        if keypoints_all:
            keypoints_xy = np.concatenate(keypoints_all, axis=0)
        else:
            keypoints_xy = np.zerps((0, 17, 2))

        if edges_all:
            edges_xy = np.stack(edges_all, axis=0)
        else:
            edges_xy = np.zeros((0, 2, 2))

        return keypoints_xy, edges_xy, edge_colors

    def draw_keypoints_on_image(self, tensor_image, keypoints):
        """
        This function will draw all the keypoints to the image
            :param tensor_image:
            :param keypoints:
            :return:
        """

        if keypoints is not None:
            # ------ preparing some sizes ------
            height, width, depth = tensor_image.shape
            aspect_ratio = float(width) / height

            # ----- configuring the plot -----
            figure, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
            figure.tight_layout(pad=0)
            ax.margins(0)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.axis("off")

            image = ax.imshow(tensor_image)
            line_segments = LineCollection([], linewidths=(4), linestyles="solid")
            ax.add_collection(line_segments)
            scat = ax.scatter([], [], s=60, color="#FF1493", zorder=3)  # TODO: add all the colors in a class

            (keypoints_locs,
             keypoints_edges,
             edge_colors) = self.keypoints_edges_display(keypoints, height, width)

            line_segments.set_segments(keypoints_edges)
            line_segments.set_colors(edge_colors)
            if keypoints_edges.shape[0]:
                line_segments.set_segments(keypoints_edges)
                line_segments.set_colors(edge_colors)
            if keypoints_locs.shape[0]:
                scat.set_offsets(keypoints_locs)

            # TODO: think about cropping the image here
            # if crop_region is not None:
            #     xmin = max(crop_region['x_min'] * width, 0.0)
            #     ymin = max(crop_region['y_min'] * height, 0.0)
            #     rec_width = min(crop_region['x_max'], 0.99) * width - xmin
            #     rec_height = min(crop_region['y_max'], 0.99) * height - ymin
            #     rect = patches.Rectangle(
            #         (xmin, ymin), rec_width, rec_height,
            #         linewidth=1, edgecolor='b', facecolor='none')
            #     ax.add_patch(rect)

            figure.canvas.draw()
            image_from_plot = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
            # image_from_plot = image_from_plot.reshape(figure.canvas.get_width_height()[::-1] + (3, ))

            plt.close(figure)

            if height is not None:
                # width = int(height / height * width)
                image_from_plot = cv2.resize(image_from_plot, (width, height), interpolation=cv2.INTER_CUBIC)

            return image_from_plot

        else:
            Messages().error("The keypoints were not generated!")
            return None


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
    KEYPOINT_EDGE_INDS_TO_COLOR = {
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
