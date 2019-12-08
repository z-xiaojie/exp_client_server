# coding='utf-8'
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil
import cv2
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
import torch
import torch.nn as nn
from threading import Lock
from _thread import *
import socket
import threading
import struct
import base64
MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
import psutil
import math
from multiprocessing import Process, Manager
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import traceback
from models import *
from utils.utils import *
from utils.datasets import *
import os
import time
import datetime
import argparse
import cv2
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np


pid = os.getpid()
py = psutil.Process(pid)

# os.sched_setaffinity(pid, {0,1,2,3,4,5,6,7})

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

gpu = True if torch.cuda.is_available() else False
lock = Lock()


def detect_image(name, response, opt, model, complex_yolo_416, Tensor):
    start_time = time.time()
    images_path = [os.path.join(opt.image_folder, name)]
    if len(images_path) == 0:
        raise Exception("no image with name {} found in {}".format(name, opt.images_path))
    print(images_path, opt.batch_size)
    images_name = os.listdir(opt.image_folder)
    images_path = [os.path.join(opt.image_folder, name) for name in images_name]
    if len(images_path) == 0:
        raise Exception("no image found in {}".format(opt.image_folder))
    # Start inference
    input_imgs = []
    image = cv2.imread(images_path[0], cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (opt.img_size, opt.img_size),
                       interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    input_imgs.append(image)
    input_imgs = np.asarray(input_imgs)
    input_imgs = torch.from_numpy(input_imgs)
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))
    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    computation_time = time.time() - start_time
    cpu = psutil.cpu_percent() / 100
    complex_yolo_416.append(computation_time * cpu * 2.8)
    print("\tyolo + {} finished in {}s, system response in {} s, cpu in {}  cycles (10^9)"
          .format(round(cpu, 3), round(computation_time, 4)
                  , round(np.average(response), 4)
                  , round(np.average(complex_yolo_416), 3)))
    # logging.info("Save all results to ./output/")
    return computation_time, cpu * 100, complex_yolo_416


class Server:
    def __init__(self, config):
        self.host = None
        self.port = None
        self.s = None
        self.c = []
        self.net = None
        self.pose = None
        self.config = config

    def run(self, port=3389):
        global pid
        host = ""
        port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((host, port))
        self.s.listen(5)
        print("start to listening...")
        while True:
            try:
                c, addr = self.s.accept()
                self.c.append(c)
                #start_new_thread(client_handler, (c, addr, self.config, pid))
                Process(target=client_handler, args=(c, addr, self.config, pid)).start()
                print("client", addr, "connected")
            except:
                self.s.close()
                break


def detect_pose(name, pose, response, config, complex_pose_438, pid):
    start = time.time()
    images_path = [os.path.join(config["images_path"], name)]
    if len(images_path) == 0:
        raise Exception("no image with name {} found in {}".format(name, config["images_path"]))
    # image_sizes = []
    for path in images_path:
        image = common.read_imgfile(path, None, None)
        humans = pose.inference(image, resize_to_default=True, upsample_size=4.0)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        # image_sizes.append(os.path.getsize(path) * 8)
    computation_time = time.time() - start
    cpu = psutil.cpu_percent() / 100
    complex_pose_438.append(computation_time * cpu * 2.8)
    print("\tpose + {} finished in {}s, system response in {} s, cpu in {} cycles(10^9)"
             .format(round(cpu, 3), round(computation_time, 4)
                      , round(np.average(response), 4)
                      , round(np.average(complex_pose_438), 3)))
    return computation_time, cpu * 100, complex_pose_438


def initial_pose_model(config):
    w, h = model_wh(config["resize"])
    if w == 0 or h == 0:
        pose = TfPoseEstimator(get_graph_path(config["pose_model"]), target_size=(432, 368))
    else:
        pose = TfPoseEstimator(get_graph_path(config["pose_model"]), target_size=(w, h))
    return pose


def initial_yolo_model(config, size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mod
    return model


def recv_msg(c):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(4, c)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    # print("mes len", msglen)
    return recvall(msglen, c)


def recvall(n, c):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = c.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def send_msg(c, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    c.sendall(msg)


def client_handler(c, addr, config, pid):
    message = {"code": 1}
    send_msg(c, json.dumps(message).encode("utf-8"))
    yolo_losses = []
    complex_yolo_416 = []
    response = []
    net = None
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    while True:
        try:
            res_start = time.time()
            data = recv_msg(c)
            start = time.time()
            info = json.loads(str(data.decode('utf-8')))
            if info["code"] == 1 and info["name"] is not None:
                if info["name"] is not None:
                    with open(config.image_folder + info["name"], 'wb') as file:
                        file.write(base64.b64decode(info["data"]))
                    if info["app"] == "yolo":
                        if net is None:
                            net = initial_yolo_model(config, info["size"])
                        compute_time, cpu, complex_yolo_416= detect_image(info["name"], response, config, net, complex_yolo_416,Tensor)
                    else:
                        if net is None:
                            net = initial_pose_model(config)
                        compute_time, cpu, complex_yolo_416 = detect_pose(info["name"], net, response, config, complex_yolo_416, pid)
                    message = {"code": 2, "time": time.time() - start, "inx": info["inx"], "cpu": cpu,
                               "compute_time": compute_time, "path": info["name"], "next": True, "timestamp": info["timestamp"]}
                    send_msg(c, json.dumps(message).encode("utf-8"))
                    response.append(round(time.time() - res_start, 3))
            if info["code"] == -1:
                c.close()
                print(addr, "close")
        except Exception as e:
            print("1." + traceback.format_exc())
            return
    print(list(response))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--image_target", type=str, default=None, help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3-tiny.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--port", type=int, default=3389, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    server = Server(opt)
    print(pid)
    server.run(port=opt.port)

