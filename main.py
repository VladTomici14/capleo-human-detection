from predictor import MoveNetPredictor
import argparse
import time
import cv2

if __name__ == "__main__":
    # ----- argparse the input image path ------
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=1, help="The video source for the program (image / video / 1 for webcam)")
    ap.add_argument("--drawing", default=True, help="Do you want to draw the keypoints on the image? (True/False)")
    ap.add_argument("--angles", default=True, help="Do you want to calculate the angles values?")
    ap.add_argument("--interpreter", default="models/lite-model_movenet_singlepose_thunder_3.tflite")
    args = vars(ap.parse_args())

    # ------- loading the model interpreter ------
    input_size = 256
    interpreter_path = str(args["interpreter"])

    # ------- setting up the predictor class ------
    predictor = MoveNetPredictor(model_path=args["interpreter"], input_size=256)
    previous_time = 0

    # ------ started to do the detection for multiple input circumstances (camera / picture / video) ------
    if args["source"] == "camera":
        # -------- webcam detection ------
        camera = cv2.VideoCapture(0)

        # ------- setting up the timer for showing FPS -----
        initial_time = time.time()

        # ------- reading from the camera frame by frame -----
        while camera.isOpened():
            ret, frame = camera.read()
            if ret:
                (height, width) = frame.shape[:2]

                # ------ making the detection using the frame -------
                output_frame = predictor.detect_on_image(frame, args["drawing"], args["angles"])

                # ------ adding FPS on the frame ------
                fps = 1 / (time.time() - previous_time)
                previous_time = time.time()
                cv2.putText(output_frame, f"FPS: {str(int(fps))}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                # ------ showing the camera feed with the results drawn -----
                cv2.imshow("Camera detection output", output_frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                break

        camera.release()

    elif cv2.imread(args["source"]) is None:
        # ------- the source file that was parsed is a video ------
        video = cv2.VideoCapture(args["source"])

        # -------- opening the video frame by frame -------
        while video.isOpened():
            ret, frame = video.read()
            if ret:
                # ------- making the detection -----
                output_frame = predictor.detect_on_image(frame, args["drawing"], args["angles"])

                # ------ showing FPS ------
                fps = 1 / (time.time() - previous_time)
                previous_time = time.time()
                cv2.putText(frame, f"FPS: {str(int(fps))}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                # ------ showing the video with the drawn results -----
                cv2.imshow("Video detection output", output_frame)
                if cv2.waitKey(1) == ord("q"):
                    break
            else:
                break

        video.release()

    else:
        # ------ the source file that was parsed is a picture ------
        input_image = cv2.imread(args["source"])

        # ------- making the detection -----
        output_image = predictor.detect_on_image(input_image, args["drawing"], args["angles"])

        # ------ showing the image with the results drawn -------
        while True:
            cv2.imshow("Picture detection output", output_image)
            if cv2.waitKey(1) == ord("q"):
                break

    cv2.destroyAllWindows()
