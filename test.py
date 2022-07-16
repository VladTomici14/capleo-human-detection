
import cv2
import argparse
from predictor import MoveNetPredictor
from main import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="path of the input image")
args = vars(ap.parse_args())

predictor = MoveNetPredictor()
interpreter = load_model("models/lite-model_movenet_singlepose_thunder_3.tflite")

image = cv2.imread(args["image"])
(height, width) = image.shape[:2]

resized = cv2.resize(image, (256, 256))
resized = predictor.detect_on_image(resized, interpreter=interpreter, input_size=256, drawing=True)

resizedback = cv2.resize(resized, (width, height))
# resizedback = predictor.detect_on_image(resizedback, interpreter=interpreter, input_size=256, drawing=True)


while True:
    cv2.imshow("Original image", image)
    cv2.imshow("Resized image(256x256)", resized)
    cv2.imshow(f"Resized image({width}x{height})", resizedback)

    if cv2.waitKey(1) == ord("q"):
        break
