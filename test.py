import argparse
import time
from threading import Thread
import cv2
import numpy as np
import sys
from queue import Queue
from imutils.video import FPS
import imutils

# TODO: think about implementing this method when reading a video
# article here : https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/

class FileVideoStream:

    def __init__(self, path, queue_size=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        self.queue = Queue(maxsize=queue_size)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            if not self.queue.full():
                ret, frame = self.stream.read()

                if not ret:
                    self.stop()
                    return

                self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def more(self):
        return self.queue.qsize() > 0

    def stop(self):
        self.stopped = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="input stream")
    args = vars(ap.parse_args())

    fvs = FileVideoStream(args["source"]).start()

    time.sleep(1)

    fps = FPS().start()

    previous_time = 0

    while fvs.more():
        frame = fvs.read()
        # frame = imutils.resize(frame, width=450)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = np.dstack([frame, frame, frame])
        # display the size of the queue on the frame
        cv2.putText(frame, "Queue Size: {}".format(fvs.queue.qsize()),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        FPS = 1 / (time.time() - previous_time)
        previous_time = time.time()
        cv2.putText(frame, f"FPS: {str(int(FPS))}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
        fps.update()
