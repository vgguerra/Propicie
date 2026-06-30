import cv2
import numpy as np
from pyorbbecsdk import Pipeline, OBFormat


def _i420_to_bgr(data, width, height):
    y = data[0:height, :]
    u = data[height : height + height // 4].reshape(height // 2, width // 2)
    v = data[height + height // 4 :].reshape(height // 2, width // 2)
    yuv = cv2.merge([y, u, v])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


def _nv12_to_bgr(data, width, height):
    y = data[0:height, :]
    uv = data[height : height + height // 2].reshape(height // 2, width)
    yuv = cv2.merge([y, uv])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)


def _nv21_to_bgr(data, width, height):
    y = data[0:height, :]
    uv = data[height : height + height // 2].reshape(height // 2, width)
    yuv = cv2.merge([y, uv])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)


def _frame_to_bgra(frame):
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if fmt == OBFormat.RGB:
        bgr = cv2.cvtColor(data.reshape((height, width, 3)), cv2.COLOR_RGB2BGR)
    elif fmt == OBFormat.BGR:
        bgr = data.reshape((height, width, 3))
    elif fmt == OBFormat.BGRA:
        return data.reshape((height, width, 4))
    elif fmt == OBFormat.RGBA:
        return cv2.cvtColor(data.reshape((height, width, 4)), cv2.COLOR_RGBA2BGRA)
    elif fmt == OBFormat.YUYV:
        bgr = cv2.cvtColor(data.reshape((height, width, 2)), cv2.COLOR_YUV2BGR_YUYV)
    elif fmt == OBFormat.MJPG:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif fmt == OBFormat.UYVY:
        bgr = cv2.cvtColor(data.reshape((height, width, 2)), cv2.COLOR_YUV2BGR_UYVY)
    elif fmt == OBFormat.I420:
        bgr = _i420_to_bgr(data, width, height)
    elif fmt == OBFormat.NV12:
        bgr = _nv12_to_bgr(data, width, height)
    elif fmt == OBFormat.NV21:
        bgr = _nv21_to_bgr(data, width, height)
    else:
        raise ValueError(f"Unsupported color format: {fmt}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)


class OrbbecCamera:
    def __init__(self):
        self.pipeline = Pipeline()
        self.pipeline.start()
        self._last_frame = None

    def has_new_color_frame(self):
        frames = self.pipeline.wait_for_frames(1000)
        if not frames:
            return False
        frame = frames.get_color_frame()
        if not frame:
            return False
        self._last_frame = _frame_to_bgra(frame)
        return True

    def get_last_color_frame(self):
        if self._last_frame is None:
            return np.zeros((1080, 1920, 4), dtype=np.uint8)
        return self._last_frame

    def close(self):
        self.pipeline.stop()
