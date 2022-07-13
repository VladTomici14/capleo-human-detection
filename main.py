from predictor import MoveNetPredictor
from matplotlib import pyplot as plt
from messages import Messages
import tensorflow as tf
from PIL import Image
import tensorflow_hub
import numpy as np
import argparse
import cv2

input_size = 256


# TODO: add comments and the pydocs
# TODO: write a function which takes one cv2 frame and returns it with the lines and dots
# TODO: make sure the code supports any image format
# TODO: add some cool gradients for the circles
# TODO: add argument for video / image / webcam

def load_model():
    """
    This function will load the model.
        :return: the final model, which can be used to make predictions
    """

    try:
        # ------- setting up the model path ------
        # thunder_model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        # module = tensorflow_hub.load(thunder_model_url)
        # interpreter = module.signatures['serving_default']

        interpreter = tf.lite.Interpreter(model_path="models/lite-model_movenet_singlepose_thunder_3.tflite")
        interpreter.allocate_tensors()

        Messages().success("The model was loaded")

        return interpreter

    except Exception:
        Messages().error(Exception)
        return None


# def detect_on_image(tensor_image, model):
#     """
#     This function can do detection for one picture
#         :param tensor_image: The image in a TF array supported format.
#         :param model: The model which we want to use
#         :return: The coordinates of the key-points if a person was found in the image.
#                  None if no one was detected in the picture.
#     """
#
#     if model is not None:
#         input_image = tf.cast(tensor_image, dtype=tf.int32)
#
#         outputs = model(input_image)
#
#         keypoints_with_scores = outputs['output_0'].numpy()
#
#         return keypoints_with_scores
#
#     else:
#         Messages().error("The model was not uploaded correctly!")
#         return None


def opencv_image_2_tensorflow(opencv_image):
    tensor_image = tf.convert_to_tensor(opencv_image, dtype=tf.float32)
    tensor_image = tf.expand_dims(tensor_image, 0)

    return tensor_image


def image2array(tensor_image):
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(tensor_image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    return input_image


def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)

    return image


if __name__ == "__main__":
    # ----- argparse the input image path ------
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=1, help="The video source for the program (image / video / 1 for webcam)")
    ap.add_argument("--draw", default=True, help="Do you want to draw the keypoints on the image? (True/False)")
    args = vars(ap.parse_args())

    interpreter = load_model()

    # loaded_image = load_image(args["image"])  # <class 'tensorflow.python.framework.ops.EagerTensor'>
    # image_array = image2array(loaded_image)
    # key_points = detect_on_image(image_array, model)
    #
    # print(len(key_points[0][0]))

    predictor = MoveNetPredictor()
    # if args["source"].find(".png") or args["source"].find(".jpg") or args["source"].find(".jpeg"):
    #     image = cv2.imread(args["source"])
    #
    #     output_image = predictor.detect_on_image(image, interpreter, input_size, drawing=True)
    #
    #     while True:
    #
    #         cv2.imshow("output image", output_image)
    #
    #         if cv2.waitKey(1) == ord("q"):
    #             break
    #
    # camera = cv2.VideoCapture(args["source"])
    # while camera.isOpened():
    #     ret, frame = camera.read()
    #
    #     if ret:
    #         output_frame = predictor.detect_on_image(frame, interpreter, input_size, drawing=True)
    #
    #         # TODO: add fps value printing
    #
    #         cv2.imshow("webcam", output_frame)
    #
    #         if cv2.waitKey(1) == ord("q"):
    #             break

    # camera.release()
    # cv2.destroyAllWindows()

    image = cv2.imread(args["source"])
    output_image = MoveNetPredictor().detect_on_image(image, interpreter, input_size, drawing=True)
    while True:
        cv2.imshow("output_frame", output_image)

        if cv2.waitKey(1) == ord("q"):
            break