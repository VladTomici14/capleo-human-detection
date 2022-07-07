from messages import Messages
import tensorflow as tf
import argparse
import cv2


def load_model(model):
    """
    This function will load the model.
        :param model: it can be the "float16" / "int8" models
        :return: the final model, which can be used to make predictions
    """

    try:
        # ------- setting up the model path ------
        input_size = 256
        if model == "float16":
            model_path = "models/lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite"

        elif model == "int8":
            model_path = "models/lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite"

        # ------- initializing the TFLite interpreter ---------
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

    except Exception:
        Messages().error("Unable to load the model. Please choose a valid one (float32 / int8)")


def detect_on_image(image_path):
    image = cv2.imread(image_path)

    pass


if __name__ == "__main__":
    # ----- argparsing the input image path ------
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path of the input image")
    ap.add_argument("--model", default="float16", help="which model to be uploaded")
    args = vars(ap.parse_args())

    load_model(args["model"])
