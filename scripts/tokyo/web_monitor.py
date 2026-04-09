"""
实时监控服务器，使用 Flask + 轮询方案。
前端每秒拉取 /api/data 获取历史数据，支持历史回放。
"""
import base64
import json
import threading
import time
from flask import Flask, render_template, jsonify
import numpy as np
import cv2
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

_history = []
_max_history = 200
_lock = threading.Lock()
_task_info = {"instruction": "", "steps": []}


def set_task_info(instruction: str, steps: list):
    """由 task_runner 调用，设置任务指令和步骤序列。"""
    global _task_info
    with _lock:
        _task_info = {"instruction": instruction, "steps": steps}


def update_state(image: np.ndarray = None, **kwargs):
    """由 task_runner 调用，记录一帧状态。"""
    frame = {
        "timestamp": time.time(),
        "time_str": time.strftime("%H:%M:%S"),
        "image": "",
        "step": 0,
        "phase": "-",
        "altitude": 0.0,
        "action": "-",
        "reasoning": "-",
        "road_visible": False,
        "road_direction": "-",
        "target_visible": False,
        "target_offset": "-",
        "distance_bucket": "-",
        "centered": False,
    }
    frame.update(kwargs)

    if image is not None:
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 65])
        frame["image"] = base64.b64encode(buf).decode("utf-8")

    with _lock:
        _history.append(frame)
        if len(_history) > _max_history:
            _history.pop(0)


def update_image(image: np.ndarray):
    """兼容旧调用。"""
    update_state(image=image)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def get_data():
    with _lock:
        # 只返回每帧的元数据，不含图像，减少传输量
        meta = [{k: v for k, v in f.items() if k != "image"} for f in _history]
    return jsonify(meta)


@app.route("/api/task_info")
def get_task_info():
    with _lock:
        return jsonify(_task_info)


@app.route("/api/frame/<int:idx>")
def get_frame(idx):
    with _lock:
        if 0 <= idx < len(_history):
            return jsonify(_history[idx])
    return jsonify({}), 404


def start_server(host="0.0.0.0", port=5000):
    import sys
    cli = sys.modules.get('flask.cli')
    if cli:
        cli.show_server_banner = lambda *x: None
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False),
        daemon=True,
    )
    t.start()
    print(f"监控界面已启动: http://localhost:{port}")
    return t
