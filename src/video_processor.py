"""
Video Processing Module for opening video streams and reading frames using OpenCV.
"""

from typing import Tuple, Optional, Union
import cv2


class VideoProcessor:
    """
    Utility class for reading video feeds (file or webcam) using OpenCV.
    """

    def __init__(self, source: Union[str, int] = 0):
        """
        :param source: Path to video file or camera index (0 for webcam).
        """
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Opens the video stream."""
        self.cap = cv2.VideoCapture(self.source)
        return self.cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Reads a single frame from the video stream."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def get_fps(self) -> float:
        """Returns frames per second of the video stream."""
        if self.cap and self.cap.isOpened():
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            return fps if fps > 0 else 30.0
        return 30.0

    def get_frame_count(self) -> int:
        """Returns total frame count for video file."""
        if self.cap and self.cap.isOpened():
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def get_dimensions(self) -> Tuple[int, int]:
        """Returns (width, height) of the video stream."""
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 0, 0

    def release(self):
        """Releases the video stream."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
