import cv2
import subprocess

class CameraController:
    """
    Encapsulates camera settings control. This class provides methods to print current
    settings, adjust contrast, brightness, focus, zoom, exposure, set default parameters,
    and perform an auto focus adjustment.
    """

    def __init__(self, cap, video_device="/dev/video2"):
        """
        Initialize the CameraController with an OpenCV VideoCapture object.
        
        :param cap: An already-opened cv2.VideoCapture object.
        :param video_device: Path to the video device (used for v4l2-ctl commands).
        """
        self.cap = cap
        self.video_device = video_device

        # Commands to set power_line_frequency (these two commands should be run each time)
        self.power_line_frequency_cmds = [
            ["v4l2-ctl", "-d", self.video_device, "--set-ctrl=power_line_frequency=0"],
            ["v4l2-ctl", "-d", self.video_device, "--set-ctrl=power_line_frequency=2"]
        ]

    def _run_power_line_frequency(self):
        """Run the necessary v4l2-ctl commands to set the camera at the proper power line frequency."""
        for cmd in self.power_line_frequency_cmds:
            subprocess.run(cmd)

    def print_current_settings(self):
        """Print out current camera settings."""
        print("******************************")
        print("Width =", self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Height =", self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Framerate =", self.cap.get(cv2.CAP_PROP_FPS))
        print("Brightness =", self.cap.get(cv2.CAP_PROP_BRIGHTNESS))
        print("Contrast =", self.cap.get(cv2.CAP_PROP_CONTRAST))
        print("Saturation =", self.cap.get(cv2.CAP_PROP_SATURATION))
        print("Gain =", self.cap.get(cv2.CAP_PROP_GAIN))
        print("Hue =", self.cap.get(cv2.CAP_PROP_HUE))
        print("Exposure =", self.cap.get(cv2.CAP_PROP_EXPOSURE))
        print("Zoom =", self.cap.get(cv2.CAP_PROP_ZOOM))
        print("Focus =", self.cap.get(cv2.CAP_PROP_FOCUS))
        print("Auto Focus =", self.cap.get(cv2.CAP_PROP_AUTOFOCUS))
        print("Auto Exposure =", self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        print("******************************")

    def process_key(self, k):
        """
        If the user presses the key associated with printing settings ('g'),
        print the current settings.
        """
        if k == ord('g'):
            self.print_current_settings()

    def adjust_contrast(self, Contrast, k):
        """
        Adjust contrast: 'y' decreases contrast; 'h' increases contrast.
        
        :param Contrast: Current contrast level.
        :param k: The key code pressed.
        :return: Updated contrast level.
        """
        if k == ord('y'):
            Contrast = max(0, Contrast - 1)
            self._run_power_line_frequency()
            if self.cap.set(cv2.CAP_PROP_CONTRAST, Contrast):
                print("Contrast is now", self.cap.get(cv2.CAP_PROP_CONTRAST))
            else:
                print("Failed to adjust contrast")
        elif k == ord('h'):
            Contrast = min(64, Contrast + 1)
            self._run_power_line_frequency()
            if self.cap.set(cv2.CAP_PROP_CONTRAST, Contrast):
                print("Contrast is now", self.cap.get(cv2.CAP_PROP_CONTRAST))
            else:
                print("Failed to adjust contrast")
        return Contrast

    def adjust_brightness(self, Brightness, k):
        """
        Adjust brightness: 'e' decreases brightness; 'd' increases brightness.
        
        :param Brightness: Current brightness value.
        :param k: The key code pressed.
        :return: Updated brightness value.
        """
        if k == ord('e'):
            Brightness = max(0, Brightness - 1)
            self._run_power_line_frequency()
            if self.cap.set(cv2.CAP_PROP_BRIGHTNESS, Brightness):
                print("Brightness is now", self.cap.get(cv2.CAP_PROP_BRIGHTNESS))
            else:
                print("Failed to adjust brightness")
        elif k == ord('d'):
            Brightness = min(64, Brightness + 1)
            self._run_power_line_frequency()
            if self.cap.set(cv2.CAP_PROP_BRIGHTNESS, Brightness):
                print("Brightness is now", self.cap.get(cv2.CAP_PROP_BRIGHTNESS))
            else:
                print("Failed to adjust brightness")
        return Brightness

    def adjust_focus(self, Focus, k):
        """
        Adjust focus: 'r' decreases focus; 'f' increases focus.
        
        :param Focus: Current focus value.
        :param k: The key code pressed.
        :return: Updated focus value.
        """
        if k == ord('r'):
            Focus = max(0, Focus - 1)
            self.cap.set(cv2.CAP_PROP_FOCUS, Focus)
            print("Focus is now", self.cap.get(cv2.CAP_PROP_FOCUS))
        elif k == ord('f'):
            Focus = min(127, Focus + 1)
            self.cap.set(cv2.CAP_PROP_FOCUS, Focus)
            print("Focus is now", self.cap.get(cv2.CAP_PROP_FOCUS))
        return Focus

    def adjust_zoom(self, zoom, k):
        """
        Adjust zoom: 'w' decreases zoom; 's' increases zoom.
        
        :param zoom: Current zoom level.
        :param k: The key code pressed.
        :return: Updated zoom level.
        """
        if k == ord('w'):
            zoom = max(0, zoom - 100)
            self.cap.set(cv2.CAP_PROP_ZOOM, zoom)
            print("Zoom is now", self.cap.get(cv2.CAP_PROP_ZOOM))
        elif k == ord('s'):
            zoom = min(16384, zoom + 100)
            self.cap.set(cv2.CAP_PROP_ZOOM, zoom)
            print("Zoom is now", self.cap.get(cv2.CAP_PROP_ZOOM))
        return zoom

    def adjust_exposure(self, Exposure, k):
        """
        Adjust exposure: 'q' decreases exposure; 'a' increases exposure.
        This method also makes calls to v4l2-ctl to set auto exposure and exposure time.
        
        :param Exposure: Current exposure value.
        :param k: The key code pressed.
        :return: Updated exposure value.
        """
        if k == ord('q'):
            Exposure = max(10, Exposure - 10)
            self._run_power_line_frequency()
            subprocess.run(["v4l2-ctl", "-d", self.video_device, "--set-ctrl=auto_exposure=3"])
            subprocess.run(["v4l2-ctl", "-d", self.video_device, "--set-ctrl=auto_exposure=1"])
            subprocess.run(["v4l2-ctl", "-d", self.video_device, f"--set-ctrl=exposure_time_absolute={Exposure}"])
            print("Exposure is now", self.cap.get(cv2.CAP_PROP_EXPOSURE))
        elif k == ord('a'):
            Exposure = min(1250, Exposure + 10)
            self._run_power_line_frequency()
            subprocess.run(["v4l2-ctl", "-d", self.video_device, "--set-ctrl=auto_exposure=3"])
            subprocess.run(["v4l2-ctl", "-d", self.video_device, "--set-ctrl=auto_exposure=1"])
            subprocess.run(["v4l2-ctl", "-d", self.video_device, f"--set-ctrl=exposure_time_absolute={Exposure}"])
            print("Exposure is now", self.cap.get(cv2.CAP_PROP_EXPOSURE))
        return Exposure

    def set_camera_settings(self):
        """
        Set default camera settings.
        """
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)  # disable auto focus for manual control
        self.cap.set(cv2.CAP_PROP_FOCUS, 34)
        self._run_power_line_frequency()
        self.cap.set(cv2.CAP_PROP_CONTRAST, 25)
        subprocess.run(["v4l2-ctl", "-d", self.video_device, "--set-ctrl=auto_exposure=3"])
        self.cap.set(cv2.CAP_PROP_ZOOM, 9000)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 32)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def auto_focus(self, focus_step: int):
        """
        Auto-adjust focus by decrementing the focus value.
        
        :param focus_step: The current focus step count.
        :return: A tuple of (updated focus_step, target_focus)
        """
        base_focus = 35
        focus_step += 1
        target_focus = base_focus - focus_step
        self.cap.set(cv2.CAP_PROP_FOCUS, target_focus)
        return focus_step, target_focus


# ------------------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit(1)

    # Initialize the camera controller using the VideoCapture object
    cam = CameraController(cap)
    cam.set_camera_settings()

    # Get initial values from the camera.
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    focus = cap.get(cv2.CAP_PROP_FOCUS)
    zoom = cap.get(cv2.CAP_PROP_ZOOM)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    focus_step = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Wait for a key press
        k = cv2.waitKey(1)
        # Process any print settings call
        cam.process_key(k)
        # Update camera properties based on user input
        contrast = cam.adjust_contrast(contrast, k)
        brightness = cam.adjust_brightness(brightness, k)
        focus = cam.adjust_focus(focus, k)
        zoom = cam.adjust_zoom(zoom, k)
        exposure = cam.adjust_exposure(exposure, k)

        # Optionally run auto focus if 'm' is pressed.
        if k == ord('m'):
            focus_step, focus = cam.auto_focus(focus_step)

        # Display the frame (for debugging/testing)
        cv2.imshow("Camera", frame)
        if k == 27:  # Esc key to quit
            break

    cap.release()
    cv2.destroyAllWindows()
