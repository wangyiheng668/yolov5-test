# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Run a Flask REST API exposing one or more YOLOv5s models."""

import argparse
import io

import torch
from flask import Flask, request
from PIL import Image

app = Flask(__name__)  # 创建一个flask应用程序
models = {}

DETECTION_URL = "/v1/object-detection/<model>"  # 定义一个URL路由，用于接收post请求，并通过<model>指定使用的模型名称。


@app.route(DETECTION_URL, methods=["POST"])  # 当接收到detection_url请求时，将执行predict函数，并且模型的指定只接受post请求
def predict(model):
    """Predict and return object detections in JSON format given an image and model name via a Flask REST API POST
    request.
    """
    if request.method != "POST":
        return

    if request.files.get("image"):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))  # 使用with语句将文件内容转换为字节流。image.open打开图像，可以确保在 读取完后自动关闭文件句柄

        # Method 2
        im_file = request.files["image"]
        im_bytes = im_file.read()   # 与with不同，这里使用read（）读取字节内容并转换成字节流，但是这种方法要手动关闭文件句柄
        im = Image.open(io.BytesIO(im_bytes))  # 但这里不需要手动关闭，因为 BytesIO 对象，它会在不再需要时自动释放内存。

        if model in models:
            results = models[model](im, size=640)  # reduce size=320 for faster inference
            return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument("--model", nargs="+", default=["yolov5s"], help="model(s) to run, i.e. --model yolov5n yolov5s")
    opt = parser.parse_args()

    for m in opt.model:
        models[m] = torch.hub.load("ultralytics/yolov5", m, force_reload=True, skip_validation=True)

    # host 指的是应用程序监听的主机地址（本地地址），0.0.0.0意味可以接收任何ip地址的请求（会在本地的所有网络接口上监听指定的端口号）。
    # port是指定应用程序监听的端口号
    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
