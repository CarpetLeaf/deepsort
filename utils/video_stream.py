import cv2

class VideoStream:
    def __init__(self, source):
        try:
            self.cap = cv2.VideoCapture(int(source))
        except:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open source {source}")

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    def release(self):
        self.cap.release()