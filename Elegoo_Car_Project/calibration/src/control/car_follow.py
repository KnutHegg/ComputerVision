import socket
import json
import time
import threading


def pack(obj) -> bytes:
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class CarFollower:
    """
    Background controller:
    - Connects to Elegoo car TCP server (IP:PORT)
    - Receives pose updates from vision (x_m, z_m, margin)
    - Sends rocker commands (FWD/BACK/LEFT/RIGHT/STOP)
    - If tag lost for LOST_SCAN_AFTER_S seconds:
        * stops
        * pans camera left-right to search with dwell when hitting ends
    """

    def __init__(
        self,
        ip="192.168.4.1",
        port=100,
        send_every=0.12,

        # --- follow tuning ---
        z_target=0.60,
        z_deadband=0.08,
        x_deadband=0.05,
        x_turn_hard=0.20,
        margin_thresh=15.0,

        # --- camera scan ---
        lost_scan_after_s=5.0,
        cam_min=40,
        cam_max=140,
        cam_start=90,
        cam_speed_deg_per_sec=70,
        cam_send_hz=20,
        cam_axis_d1=1,

        # dwell after direction flip
        cam_flip_pause_s=2.0,
    ):
        self.IP = ip
        self.PORT = port
        self.SEND_EVERY = float(send_every)

        self.Z_TARGET = float(z_target)
        self.Z_DB = float(z_deadband)
        self.X_DB = float(x_deadband)
        self.X_HARD = float(x_turn_hard)
        self.MARGIN_THRESH = float(margin_thresh)

        self.LOST_SCAN_AFTER_S = float(lost_scan_after_s)
        self.CAM_MIN = int(cam_min)
        self.CAM_MAX = int(cam_max)
        self.CAM_START = float(cam_start)
        self.CAM_SPEED = float(cam_speed_deg_per_sec)
        self.CAM_SEND_HZ = float(cam_send_hz)
        self.CAM_AXIS_D1 = int(cam_axis_d1)
        self.CAM_FLIP_PAUSE_S = float(cam_flip_pause_s)

        # latest vision update
        self._lock = threading.Lock()
        self._last_seen_t = None
        self._x = None
        self._z = None
        self._margin = None

        # connection
        self.sock = None

        # camera scan state
        self._cam_angle = float(self.CAM_START)
        self._cam_dir = +1.0
        self._last_cam_send = 0.0
        self._last_cam_update = time.time()
        self._scan_pause_until = 0.0  # dwell timer

        # drive state
        self._last_drive_send = 0.0
        self._last_cmd_label = None

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---- protocol helpers ----
    def rocker(self, direction_code: int):
        # N=102 rocker mode, D1: 1 fwd, 2 back, 3 left, 4 right
        return {"H": 1, "N": 102, "D1": int(direction_code)}

    def stop_cmd(self):
        return {"H": 1, "N": 100}

    def cam_pan(self, angle: int):
        # N=5 servo; D1 selects axis, D2 is angle
        return {"H": 1, "N": 5, "D1": int(self.CAM_AXIS_D1), "D2": int(angle)}

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((self.IP, self.PORT))
        s.settimeout(0.05)
        try:
            _ = s.recv(256)
        except Exception:
            pass
        return s

    def _drain(self, sock):
        try:
            while True:
                data = sock.recv(4096)
                if not data:
                    return False
        except socket.timeout:
            return True
        except Exception:
            return False

    def _send(self, obj):
        self.sock.sendall(pack(obj))

    # ---- public API from vision ----
    def update(self, x_m, z_m, margin):
        """
        Call this from vision when you have a pose.
        x_m: left/right meters (x>0 = tag to the RIGHT of camera)
        z_m: forward distance meters
        margin: decision_margin from apriltags
        """
        now = time.time()
        with self._lock:
            self._x = None if x_m is None else float(x_m)
            self._z = None if z_m is None else float(z_m)
            self._margin = None if margin is None else float(margin)

            if (
                self._x is not None and
                self._z is not None and
                self._margin is not None and
                self._margin >= self.MARGIN_THRESH
            ):
                self._last_seen_t = now

    def close(self):
        self._stop_event.set()
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self.sock is not None:
                self._send(self.stop_cmd())
        except Exception:
            pass
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass

    # ---- control logic ----
    def _choose_drive_command(self, x, z):
        """
        Returns (label, cmd_dict)
        - If lateral error is big, turn in place first to avoid 'spinning while backing'
        - Otherwise move toward Z_TARGET and sprinkle in turns
        """
        z_err = z - self.Z_TARGET

        # Big lateral error -> prioritize turning in place
        if abs(x) >= self.X_HARD:
            if x > 0:
                return ("RIGHT", self.rocker(4))
            else:
                return ("LEFT", self.rocker(3))

        # Decide forward/back/stop from distance
        if abs(z_err) <= self.Z_DB:
            move = "STOP"
        elif z_err > 0:
            move = "FWD"
        else:
            move = "BACK"

        # Decide turn direction
        if abs(x) <= self.X_DB:
            turn = "NONE"
        elif x > 0:
            turn = "RIGHT"
        else:
            turn = "LEFT"

        # If not moving, just turn a bit to center
        if move == "STOP":
            if turn == "LEFT":
                return ("LEFT", self.rocker(3))
            if turn == "RIGHT":
                return ("RIGHT", self.rocker(4))
            return ("STOP", self.stop_cmd())

        # Moving: alternate between move and turn (simple PWM)
        # keep turn probability modest to avoid "spin"
        p_turn = min(abs(x) / self.X_HARD, 1.0) * 0.35  # <= 35% of time turning

        cycle = 0.6
        phase = (time.time() % cycle) / cycle
        do_turn = (turn != "NONE") and (phase < p_turn)

        if do_turn:
            if turn == "LEFT":
                return ("LEFT", self.rocker(3))
            if turn == "RIGHT":
                return ("RIGHT", self.rocker(4))

        if move == "FWD":
            return ("FWD", self.rocker(1))
        else:
            return ("BACK", self.rocker(2))

    def _camera_scan_step(self):
        now = time.time()

        # dwell after direction flip
        if now < self._scan_pause_until:
            # still send current angle occasionally (keeps servo alive)
            cam_period = 1.0 / self.CAM_SEND_HZ
            if (now - self._last_cam_send) >= cam_period:
                self._send(self.cam_pan(int(self._cam_angle)))
                self._last_cam_send = now
            return

        dt = now - self._last_cam_update
        self._last_cam_update = now

        self._cam_angle += self._cam_dir * self.CAM_SPEED * dt

        flipped = False
        if self._cam_angle >= self.CAM_MAX:
            self._cam_angle = float(self.CAM_MAX)
            self._cam_dir = -1.0
            flipped = True
        elif self._cam_angle <= self.CAM_MIN:
            self._cam_angle = float(self.CAM_MIN)
            self._cam_dir = +1.0
            flipped = True

        if flipped:
            self._scan_pause_until = now + self.CAM_FLIP_PAUSE_S

        cam_period = 1.0 / self.CAM_SEND_HZ
        if (now - self._last_cam_send) >= cam_period:
            self._send(self.cam_pan(int(self._cam_angle)))
            self._last_cam_send = now

    def _loop(self):
        # connect (retry forever)
        while not self._stop_event.is_set():
            try:
                self.sock = self._connect()
                break
            except Exception:
                time.sleep(0.5)

        if self.sock is None:
            return

        # set camera to center initially
        try:
            self._send(self.cam_pan(int(self._cam_angle)))
        except Exception:
            pass

        while not self._stop_event.is_set():
            # keep connection healthy
            if not self._drain(self.sock):
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None

                # reconnect
                while not self._stop_event.is_set():
                    try:
                        self.sock = self._connect()
                        # reapply camera
                        self._send(self.cam_pan(int(self._cam_angle)))
                        self._last_cmd_label = None
                        self._last_drive_send = 0.0
                        self._last_cam_send = 0.0
                        self._last_cam_update = time.time()
                        self._scan_pause_until = 0.0
                        break
                    except Exception:
                        time.sleep(0.5)

                if self.sock is None:
                    continue

            now = time.time()
            with self._lock:
                last_seen = self._last_seen_t
                x = self._x
                z = self._z
                margin = self._margin

            # If tag lost long enough => stop + scan camera
            if (last_seen is None) or ((now - last_seen) > self.LOST_SCAN_AFTER_S):
                try:
                    self._send(self.stop_cmd())
                except Exception:
                    pass
                try:
                    self._camera_scan_step()
                except Exception:
                    pass
                time.sleep(0.01)
                continue

            # Tag recently seen, but if pose not valid/confident => stop
            if x is None or z is None or margin is None or margin < self.MARGIN_THRESH:
                cmd_label, cmd = ("STOP", self.stop_cmd())
            else:
                cmd_label, cmd = self._choose_drive_command(x, z)

            # send at controlled rate
            if (now - self._last_drive_send) >= self.SEND_EVERY or cmd_label != self._last_cmd_label:
                try:
                    self._send(cmd)
                    self._last_drive_send = now
                    self._last_cmd_label = cmd_label
                except Exception:
                    try:
                        self.sock.close()
                    except Exception:
                        pass
                    self.sock = None

            time.sleep(0.005)