# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Perform test request."""
# 客户端像API接口发送数据的请求
import pprint

import requests

DETECTION_URL = "http://localhost:5000/v1/object-detection/yolov5s"  # 这里是指向目标检测的API的具体路径
IMAGE = "zidane.jpg"

# Read image
with open(IMAGE, "rb") as f:
    image_data = f.read()

response = requests.post(DETECTION_URL, files={"image": image_data}).json()  # 服务器返回的响应存储在response中

pprint.pprint(response)
