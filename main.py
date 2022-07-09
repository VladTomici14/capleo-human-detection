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

def load_model():
    """
    This function will load the model.
        :return: the final model, which can be used to make predictions
    """

    try:
        # ------- setting up the model path ------
        thunder_model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        module = tensorflow_hub.load(thunder_model_url)
        model = module.signatures['serving_default']

        Messages().success("The model was loaded")

        return model

    except Exception:
        Messages().error(Exception)
        return None


def detect_on_image(tensor_image, model):
    """
    This function can do detection for one picture
        :param tensor_image: The image in a TF array supported format.
        :param model: The model which we want to use
        :return: The coordinates of the key-points if a person was found in the image.
                 None if no one was detected in the picture.
    """

    if model is not None:
        input_image = tf.cast(tensor_image, dtype=tf.int32)

        outputs = model(input_image)

        keypoints_with_scores = outputs['output_0'].numpy()

        return keypoints_with_scores

    else:
        Messages().error("The model was not uploaded correctly!")
        return None


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
    ap.add_argument("--image", required=True, help="path of the input image")
    ap.add_argument("--draw", default=True, help="Do you want to draw the keypoints on the image? (True/False)")
    args = vars(ap.parse_args())

    model = load_model()

    # loaded_image = load_image(args["image"])  # <class 'tensorflow.python.framework.ops.EagerTensor'>
    # image_array = image2array(loaded_image)
    # key_points = detect_on_image(image_array, model)
    #
    # print(len(key_points[0][0]))

    # FIXME : problems with loading and processing the images

    if args["draw"]:
        Messages().success("Accessed the recognition part")

        # predictor = MoveNetPredictor()
        # output_image_array = predictor.draw_keypoints_on_image(loaded_image, key_points)  # <class 'numpy.ndarray'>
        # output_image = Image.fromarray(output_image_array)
        # output_image.save("test.png")

        # from matplotlib import pyplot as plt
        # plt.imshow(output_image_array, interpolation="nearest")
        # plt.show()

    # Load the input image.
    image_path = args["image"]
    image = tf.io.read_file(image_path) # TODO: make sure that it supports more photo formats
    image = tf.image.decode_jpeg(image)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    # Run model inference.
    keypoints_with_scores = detect_on_image(input_image, model)

    # Visualize the predictions with image.
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(display_image, input_size, input_size), dtype=tf.int32)
    output_overlay = MoveNetPredictor().draw_keypoints_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

    print(type(output_overlay))

    plt.figure(figsize=(5, 5))
    plt.imshow(output_overlay)
    _ = plt.axis('off')
    plt.show()
