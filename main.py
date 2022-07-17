from predictor import MoveNetPredictor, Keypoint
from matplotlib import pyplot as plt
from messages import Messages
import tensorflow as tf
from PIL import Image
import tensorflow_hub
import numpy as np
import argparse
import time
import cv2

# TODO: add comments and the documentation
# TODO: make sure the code supports any image format
# TODO: add some cool gradients for the circles

if __name__ == "__main__":
    # ----- argparse the input image path ------
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=1, help="The video source for the program (image / video / 1 for webcam)")
    ap.add_argument("--drawing", default=True, help="Do you want to draw the keypoints on the image? (True/False)")
    ap.add_argument("--interpreter", default="models/lite-model_movenet_singlepose_thunder_3.tflite")
    args = vars(ap.parse_args())

    # ------- loading the model interpreter ------
    input_size = 256
    interpreter_path = str(args["interpreter"])

    # ------- setting up the predictor class ------
    predictor = MoveNetPredictor()
    previous_time = 0

    # ------ started to do the detection for multiple input circumstances (camera / picture / video) ------
    if args["source"] == "camera":
        # -------- webcam detection ------
        camera = cv2.VideoCapture(0)
        initial_time = time.time()

        while camera.isOpened():
            ret, frame = camera.read()
            if ret:
                (height, width) = frame.shape[:2]
                cropped_camera = predictor.crop_camera(frame)

                output_frame = frame.copy()
                output_frame = predictor.detect_on_image(output_frame, interpreter_path, input_size, drawing=args["drawing"])

                # resized_frame = cv2.resize(frame, (input_size, input_size))
                # resized_frame = predictor.detect_on_image(resized_frame, interpreter_path, input_size, drawing=args["drawing"])
                # resized_frame = cv2.resize(resized_frame, (width, height))
                #
                # square_camera = cv2.resize(frame, (input_size, input_size))
                # square_camera = predictor.detect_on_image(square_camera, interpreter_path, input_size, drawing=args["drawing"])

                # angle = predictor.calculate_angle(point1=Keypoint.KEYPOINT_DICT["right_shoulder"],
                #                                   connection_point=Keypoint.KEYPOINT_DICT["right_elbow"],
                #                                   point2=Keypoint.KEYPOINT_DICT["right_wrist"])
                #
                # print(angle)

                # ------ showing FPS ------
                fps = 1 / (time.time() - previous_time)
                previous_time = time.time()
                cv2.putText(cropped_camera, f"FPS: {str(int(fps))}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                cv2.imshow("Original camera", output_frame)
                # cv2.imshow("Resized camera", resized_frame)
                # cv2.imshow("Square sized camera", square_camera)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                break

        camera.release()

    elif cv2.imread(args["source"]) is None:
        # ------- the source file that was parsed is a video ------
        video = cv2.VideoCapture(args["source"])
        while video.isOpened():
            ret, frame = video.read()

            if ret:
                output_frame = predictor.detect_on_image(frame, interpreter_path, input_size, drawing=args["drawing"])

                # ------ showing FPS ------
                fps = 1 / (time.time() - previous_time)
                previous_time = time.time()
                cv2.putText(frame, f"FPS: {str(int(fps))}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                # ------ showing the video -----
                cv2.imshow("Video detection output", output_frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                break

        video.release()

    else:
        # ------ the source file that was parsed is a picture ------
        input_image = cv2.imread(args["source"])

        output_image = predictor.detect_on_image(input_image, interpreter_path, input_size, drawing=args["drawing"])

        angle = predictor.calculate_angle(point1=Keypoint.KEYPOINT_DICT["right_shoulder"],
                                          connection_point=Keypoint.KEYPOINT_DICT["right_elbow"],
                                          point2=Keypoint.KEYPOINT_DICT["right_wrist"])

        print(angle)

        while True:
            cv2.imshow("Picture detection output", output_image)
            if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()
