"""
ESP32 Microcontroller Communication Module (Placeholder).
Future module to send dynamic signal timing commands to physical ESP32 hardware via Serial / Wi-Fi.
"""


class ESP32Communicator:
    """
    Placeholder class for sending timing data to ESP32 signal lights.
    """

    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate

    def send_timing(self, lane_id: str, state: str, duration_sec: int):
        """
        Send timing parameters to ESP32 board.
        Will be implemented in hardware integration step.
        """
        raise NotImplementedError("ESP32 hardware communication will be implemented in hardware phase.")
