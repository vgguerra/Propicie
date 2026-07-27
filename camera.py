import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Context, OBLogLevel


def _disable_sdk_logging():
    ctx = Context()
    ctx.set_logger_to_console(OBLogLevel.NONE)
    ctx.set_logger_to_file(OBLogLevel.NONE, "")


def _mjpg_to_bgra(data):
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)


class OrbbecCamera:
    def __init__(self):
        _disable_sdk_logging()
        self.pipeline = Pipeline()
        self.pipeline.start()
        self._last_frame = np.zeros((1080, 1920, 4), dtype=np.uint8)

    def has_new_color_frame(self):
        frames = self.pipeline.wait_for_frames(1000)
        if not frames:
            return False
        frame = frames.get_color_frame()
        if not frame:
            return False
        self._last_frame = _mjpg_to_bgra(np.asanyarray(frame.get_data()))
        return True

    def get_last_color_frame(self):
        return self._last_frame

    def close(self):
        self.pipeline.stop()
