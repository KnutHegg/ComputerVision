import time
import requests
import cv2
import numpy as np


def mjpeg_frames(url: str, connect_timeout=5, read_timeout=5, retry_delay=0.5):
    """
    Yields BGR frames from an MJPEG HTTP stream.
    Auto-reconnects on timeouts/disconnects (common with ESP32 streams).
    """
    while True:
        try:
            # timeout can be (connect_timeout, read_timeout)
            r = requests.get(url, stream=True, timeout=(connect_timeout, read_timeout))
            r.raise_for_status()

            buffer = b""
            for chunk in r.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                buffer += chunk

                a = buffer.find(b"\xff\xd8")  # JPEG start
                b = buffer.find(b"\xff\xd9")  # JPEG end
                if a != -1 and b != -1 and b > a:
                    jpg = buffer[a:b + 2]
                    buffer = buffer[b + 2:]

                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        yield frame

        except (requests.exceptions.RequestException, TimeoutError) as e:
            # Stream stalled / wifi hiccup / ESP32 rebooted
            print(f"[mjpeg] stream error: {type(e).__name__}: {e}  -> reconnecting...")
            time.sleep(retry_delay)
            continue