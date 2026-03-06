import socket
import json
import time
import os
import keyboard

IP = os.getenv("CAR_IP", "192.168.4.1")
PORT = int(os.getenv("CAR_PORT", "100"))

# Driving
SEND_EVERY = 0.12  # resend while holding drive key (keeps it responsive)

# Camera (hold-to-move)
CAM_MIN = 0
CAM_MAX = 180
CAM_START = 90
CAM_SPEED_DEG_PER_SEC = 90   # how fast camera pans while holding Q/E
CAM_SEND_HZ = 25             # how often to send servo updates while holding
CAM_AXIS_D1 = 1              # 1 is commonly pan; set to 2 if your tilt/pan are swapped

def pack(obj) -> bytes:
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")

def connect():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect((IP, PORT))
    s.settimeout(1)
    try:
        first = s.recv(256)
        print("Connected. First recv:", first, flush=True)
    except Exception:
        print("Connected.", flush=True)
    s.settimeout(0.05)
    return s

def drain(sock):
    try:
        while True:
            data = sock.recv(4096)
            if not data:
                return False
    except socket.timeout:
        return True
    except Exception:
        return False

def send(sock, obj, label=None):
    sock.sendall(pack(obj))
    if label:
        print(">>", label, obj, flush=True)

# --- Commands ---
def rocker(direction_code: int):
    # N=102 rocker mode, D1: 1 fwd, 2 back, 3 left, 4 right
    return {"H": 1, "N": 102, "D1": int(direction_code)}

def stop_cmd():
    return {"H": 1, "N": 100}

def cam_pan(angle: int):
    # N=5 servo; D1 selects axis, D2 is angle
    return {"H": 1, "N": 5, "D1": int(CAM_AXIS_D1), "D2": int(angle)}

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def main():
    print("WASD drive | SPACE stop | hold Q/E camera pan | ESC quit", flush=True)
    sock = connect()

    # Camera state
    cam_angle = float(CAM_START)
    last_cam_send = 0.0
    last_cam_update = time.time()

    # Send initial camera position once
    send(sock, cam_pan(int(cam_angle)), "CAM_INIT")

    last_drive_send = 0.0
    last_state = None

    cam_send_period = 1.0 / CAM_SEND_HZ

    try:
        while True:
            if not drain(sock):
                try:
                    sock.close()
                except Exception:
                    pass
                sock = connect()
                send(sock, cam_pan(int(cam_angle)), "CAM_REAPPLY")
                last_state = None
                last_drive_send = 0.0
                last_cam_send = 0.0
                last_cam_update = time.time()

            if keyboard.is_pressed("esc"):
                send(sock, stop_cmd(), "STOP+QUIT")
                break

            if keyboard.is_pressed("space"):
                send(sock, stop_cmd(), "STOP")
                time.sleep(0.05)
                continue

            # --- Camera: continuous while holding ---
            now = time.time()
            dt = now - last_cam_update
            last_cam_update = now

            # IMPORTANT: your servo direction is inverted relative to our assumption.
            # So: Q should turn LEFT => INCREASE angle
            #     E should turn RIGHT => DECREASE angle
            cam_dir = 0
            if keyboard.is_pressed("q"):
                cam_dir += 1   # <-- flipped
            if keyboard.is_pressed("e"):
                cam_dir -= 1   # <-- flipped

            if cam_dir != 0:
                cam_angle = clamp(
                    cam_angle + cam_dir * CAM_SPEED_DEG_PER_SEC * dt,
                    CAM_MIN,
                    CAM_MAX
                )
                if (now - last_cam_send) >= cam_send_period:
                    send(sock, cam_pan(int(cam_angle)))
                    last_cam_send = now

            # --- Driving ---
            if keyboard.is_pressed("w"):
                state = ("FWD", rocker(1))
            elif keyboard.is_pressed("s"):
                state = ("BACK", rocker(2))
            elif keyboard.is_pressed("a"):
                state = ("LEFT", rocker(3))
            elif keyboard.is_pressed("d"):
                state = ("RIGHT", rocker(4))
            else:
                state = ("STOP", stop_cmd())

            now = time.time()

            if state != last_state:
                send(sock, state[1], state[0])
                last_state = state
                last_drive_send = now
            elif (now - last_drive_send) >= SEND_EVERY and state[0] != "STOP":
                send(sock, state[1])
                last_drive_send = now

            time.sleep(0.005)

    finally:
        try:
            send(sock, stop_cmd(), "STOP")
        except Exception:
            pass
        try:
            sock.close()
        except Exception:
            pass
        print("Disconnected.", flush=True)

if __name__ == "__main__":
    main()
