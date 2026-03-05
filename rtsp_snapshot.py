import subprocess
import numpy as np
import cv2


def rtsp_snapshot_bgr(rtsp_url: str, timeout_s: int = 8) -> np.ndarray:
    """
    Делает один кадр через ffmpeg и возвращает BGR numpy array.
    ВАЖНО: ffmpeg должен быть установлен.
    """
    # сначала узнаем размер потока быстро (ffprobe не используем для скорости)
    # читаем 1 кадр в MJPEG и декодим через OpenCV
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout_s)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(f"ffmpeg snapshot failed rc={p.returncode}")
    data = np.frombuffer(p.stdout, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode returned None")
    return img
