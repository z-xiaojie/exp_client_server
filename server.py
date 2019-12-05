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
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
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
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.utils import non_max_suppression, bbox_iou
import psutil
import math

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

pid = os.getpid()
py = psutil.Process(pid)

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

gpu = True if torch.cuda.is_available() else False
lock = Lock()


class Server:
    def __init__(self, config):
        self.host = None
        self.port = None
        self.s = None
        self.c = []
        self.net = None
        self.pose = None
        self.config = config
        self.classes = open(self.config["classes_names_path"], "r").read().split("\n")[:-1]
        try:
            self.initial_yolo_model()
            self.initial_pose_model()
        except Exception as e:
            print(e.__str__())
            exit(1)
        self.complex_yolo_416 = []
        self.complex_pose_438 = []

    def initial_pose_model(self):
        w, h = model_wh(self.config["resize"])
        if w == 0 or h == 0:
            self.pose = TfPoseEstimator(get_graph_path(self.config["pose_model"]), target_size=(432, 368))
        else:
            self.pose = TfPoseEstimator(get_graph_path(self.config["pose_model"]), target_size=(w, h))

    def initial_yolo_model(self):
        is_training = False
        # Load and initialize network
        self.net = ModelMain(config, is_training=is_training)
        self.net.train(is_training)

        # Set data parallel
        self.net = nn.DataParallel(self.net)
        if gpu:
            self.net = self.net.cuda()

        # Restore pretrain model
        if self.config["pretrain_snapshot"]:
            logging.info("load checkpoint from {}".format(self.config["pretrain_snapshot"]))
            if gpu:
                state_dict = torch.load(config["pretrain_snapshot"])
            else:
                state_dict = torch.load(config["pretrain_snapshot"], map_location=torch.device('cpu'))
            self.net.load_state_dict(state_dict)
        else:
            raise Exception("missing pretrain_snapshot!!!")

        # YOLO loss with 3 scales
        self.yolo_losses = []
        for i in range(3):
            self.yolo_losses.append(YOLOLoss(self.config["yolo"]["anchors"][i],
                                             self.config["yolo"]["classes"], (self.config["img_w"], self.config["img_h"])))

    def detect_image(self, name, response):
        start_time = time.time()
        images_path = [os.path.join(self.config["images_path"], name)]
        if len(images_path) == 0:
            raise Exception("no image with name {} found in {}".format(name, self.config["images_path"]))

        # Start inference
        batch_size = self.config["batch_size"]
        for step in range(0, len(images_path), batch_size):
            # preprocess
            images = []
            images_origin = []
            image_sizes = []
            for path in images_path[step * batch_size: (step + 1) * batch_size]:
                logging.info("processing: {}".format(path))
                image = cv2.imread(path, cv2.IMREAD_COLOR)
                # image_sizes.append(os.path.getsize(path) * 8)
                if image is None:
                    logging.error("read path error: {}. skip it.".format(path))
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images_origin.append(image)  # keep for save result
                image = cv2.resize(image, (self.config["img_w"], self.config["img_h"]),
                                   interpolation=cv2.INTER_LINEAR)
                image = image.astype(np.float32)
                image /= 255.0
                image = np.transpose(image, (2, 0, 1))
                image = image.astype(np.float32)
                images.append(image)
            images = np.asarray(images)
            images = torch.from_numpy(images)

            lock.acquire()
            # inference
            with torch.no_grad():
                outputs = self.net(images)
                output_list = []
                for i in range(3):
                    output_list.append(self.yolo_losses[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                       conf_thres=self.config["confidence_threshold"],
                                                       nms_thres=0.45)
            lock.release()
            # write result images. Draw bounding boxes and labels of detections
            if not os.path.isdir("./output/"):
                os.makedirs("./output/")
            for idx, detections in enumerate(batch_detections):
                plt.figure()
                fig, ax = plt.subplots(1)
                ax.imshow(images_origin[idx])
                if detections is not None:
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    bbox_colors = random.sample(colors, n_cls_preds)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                        # Rescale coordinates to original dimensions
                        ori_h, ori_w = images_origin[idx].shape[:2]
                        pre_h, pre_w = self.config["img_h"], self.config["img_w"]
                        box_h = ((y2 - y1) / pre_h) * ori_h
                        box_w = ((x2 - x1) / pre_w) * ori_w
                        y1 = (y1 / pre_h) * ori_h
                        x1 = (x1 / pre_w) * ori_w
                        # Create a Rectangle patch
                        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                                 edgecolor=color,
                                                 facecolor='none')
                        # Add the bbox to the plot
                        ax.add_patch(bbox)
                        # Add label
                        plt.text(x1, y1, s=self.classes[int(cls_pred)], color='white',
                                 verticalalignment='top',
                                 bbox={'color': color, 'pad': 0})
                # Save generated image with detections
                plt.axis('off')
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                plt.savefig('output/{}'.format(name), bbox_inches='tight', pad_inches=0.0)
                plt.close()
        computation_time = time.time() - start_time
        cpu = py.cpu_percent() / 100
        self.complex_yolo_416.append(computation_time * cpu * 2.8)
        print("\tyolo + {} finished in {}s, system response in {} s, cpu in {}  cycles (10^9)"
              .format(round(cpu,3), round(computation_time,4)
                      , round(np.average(response),4)
                      , round(np.average(self.complex_yolo_416), 3)))
        # logging.info("Save all results to ./output/")
        return computation_time

    def detect_pose(self, name, response):
        start = time.time()
        images_path = [os.path.join(self.config["images_path"], name)]
        if len(images_path) == 0:
            raise Exception("no image with name {} found in {}".format(name, self.config["images_path"]))
        image_sizes = []
        for path in images_path:
            image = common.read_imgfile(path, None, None)
            humans = self.pose.inference(image, resize_to_default=True, upsample_size=4.0)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            # image_sizes.append(os.path.getsize(path) * 8)
        computation_time = time.time() - start
        cpu = py.cpu_percent() / 100
        self.complex_pose_438.append(computation_time * cpu * 2.8)
        print("\tpose + {} finished in {}s, system response in {} s, cpu in {} cycles(10^9)"
              .format(round(cpu, 3), round(computation_time, 4)
                      , round(np.average(response), 4)
                      , round(np.average(self.complex_pose_438), 3)))
        return computation_time

    def run(self, port=3389):
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
                start_new_thread(self.client_threaded, (c, addr))
                print("client", addr, "connected")
            except:
                self.s.close()
                break

    def recv_msg(self, c):
        # Read message length and unpack it into an integer
        raw_msglen = self.recvall(4, c)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Read the message data
        # print("mes len", msglen)
        return self.recvall(msglen, c)

    def recvall(self, n, c):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = c.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def send_msg(self, c, msg):
        # Prefix each message with a 4-byte length (network byte order)
        msg = struct.pack('>I', len(msg)) + msg
        c.sendall(msg)

    def client_threaded(self, c, addr):
        message = {"code": 1}
        self.send_msg(c, json.dumps(message).encode("utf-8"))
        response = []
        while True:
            try:
                res_start = time.time()
                data = self.recv_msg(c)
                start = time.time()
                info = json.loads(str(data.decode('utf-8')))
                # logging.info(start, info["start"])
                if info["code"] == 1 and info["name"] is not None:
                    path = info["name"]
                    if path is not None:
                        with open(self.config["images_path"] + path, 'wb') as file:
                            file.write(base64.b64decode(info["data"]))
                        if info["app"] == "yolo":
                            compute_time = self.detect_image(path, response)
                        else:
                            compute_time = self.detect_pose(path, response)
                        message = {"code": 2, "time": time.time() - start, "inx": info["inx"],
                                   "compute_time": compute_time, "path": path, "next": True, "timestamp": info["timestamp"]}
                        self.send_msg(c, json.dumps(message).encode("utf-8"))
                        response.append(round(time.time() - res_start, 3))
                if info["code"] == -1:
                    c.close()
                    print(addr, "close")
                    print(list(response))
            except Exception as e:
                print(list(response))
                return


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR,
                        format="[%(asctime)s %(filename)s] %(message)s")
    if len(sys.argv) != 2:
        logging.error("Usage: python test_images.py params.py")
        sys.exit()
    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    server = Server(config)
    print(pid)
    server.run(port=sys.argv[2])

