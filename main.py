import numpy as np
import datetime
import sys
import os
import shutil
import struct
import subprocess
import time
from subprocess import CalledProcessError
import threading
from multiprocessing import Lock, Pool, cpu_count

# Locks
stdstream_lock = Lock()
logging_lock = Lock()

# Application Constants
VERSION = "4.0.0"
VERSION_STRING = "crunch v" + VERSION

# Processor Constant
#  - Modify this to an integer value if you want to fix the number of
#    processes spawned during execution.  The process number is
#    automatically defined during source execution when this is defined
#    as a value of 0
PROCESSES = 0

# Dependency Path Constants for Command Line Executable
#  - Redefine these path strings to use system-installed versions of
#    pngquant and zopflipng (e.g. to "/usr/local/bin/[executable]")
PNGQUANT_CLI_PATH = os.path.join(os.path.expanduser("~"), "pngquant", "pngquant")
ZOPFLIPNG_CLI_PATH = os.path.join(os.path.expanduser("~"), "zopfli", "zopflipng")

# Crunch Directory (dot directory in $HOME)
CRUNCH_DOT_DIRECTORY = os.path.join(os.path.expanduser("~"), ".crunch")

# Log File Path Constants
LOGFILE_PATH = os.path.join(CRUNCH_DOT_DIRECTORY, "crunch.log")

HELP_STRING = """
==================================================
 crunch
  Copyright 2019 Christopher Simpkins
  MIT License

  Source: https://github.com/chrissimpkins/Crunch
==================================================

crunch is a command line executable that performs lossy optimization of one or more png image files with pngquant and zopflipng.

Usage:
    $ crunch [image path 1]...[image path n]

Options:
    --help, -h      application help
    --usage         application usage
    --version, -v   application version
"""

USAGE = "$ crunch [image path 1]...[image path n]"

avg_processing_time = []
avg_post_file_size = []
avg_reduce_ratio = []

avg_size = []
avg_complexity_yolo = []
cpu_utilization = []
response = []
rate = []
compute_time = []
transmission_time = []
current_img_path_inx = 0
release = True

# ///////////////////////
# FUNCTION DEFINITIONS
# ///////////////////////

pre = 0

def optimize_png(inx, png_path, quality, s_quality, speed, save_dir):
    global avg_processing_time, avg_post_file_size, pre

    start = time.time()
    img = ImageFile(png_path)

    # ////////////////////////
    # ANSI COLOR DEFINITIONS
    # ////////////////////////
    if not is_gui(sys.argv):
        ERROR_STRING = "[ " + format_ansi_red("!") + " ]"
    else:
        ERROR_STRING = "[ ! ]"

    PNGQUANT_EXE_PATH = get_pngquant_path()
    ZOPFLIPNG_EXE_PATH = get_zopflipng_path()

    # --------------
    # pngquant stage
    # --------------
    pngquant_options = (
        " --quality=" + quality + " --skip-if-larger --force --strip --speed " + speed + " --ext -crunch.png "
    )
    pngquant_command = (
        PNGQUANT_EXE_PATH + pngquant_options + shellquote(img.pre_filepath)
    )
    try:
        subprocess.check_output(pngquant_command, stderr=subprocess.STDOUT, shell=True)
    except CalledProcessError as cpe:
        if cpe.returncode == 98:
            # this is the status code when file size increases with execution of pngquant.
            # ignore at this stage, original file copied at beginning of zopflipng processing
            # below if it is not present due to these errors
            pass
        elif cpe.returncode == 99:
            # this is the status code when the image quality falls below the set min value
            # ignore at this stage, original lfile copied at beginning of zopflipng processing
            # below if it is not present to these errors
            pass
        else:
            stdstream_lock.acquire()
            sys.stderr.write(
                ERROR_STRING
                + " "
                + img.pre_filepath
                + " processing failed at the pngquant stage."
                + os.linesep
            )
            stdstream_lock.release()
            if is_gui(sys.argv):
                log_error(
                    img.pre_filepath
                    + " processing failed at the pngquant stage. "
                    + os.linesep
                    + str(cpe)
                )
                return None
            else:
                raise cpe
    except Exception as e:
        if is_gui(sys.argv):
            log_error(
                img.pre_filepath
                + " processing failed at the pngquant stage. "
                + os.linesep
                + str(e)
            )
            return None
        else:
            raise e
    avg_processing_time.append(time.time() - start)
    # ---------------
    # zopflipng stage
    # ---------------
    # use --filters=0 by default for quantized PNG files (based upon testing by CS)
    zopflipng_options = " -y --filters=0 "
    # confirm that a file with proper path was generated by pngquant
    # pngquant does not write expected file path if the file was larger after processing
    if not os.path.exists(img.post_filepath):
        shutil.copy(img.pre_filepath, img.post_filepath)
        # If pngquant did not quantize the file, permit zopflipng to attempt compression with mulitple
        # filters.  This achieves better compression than the default approach for non-quantized PNG
        # files, but takes significantly longer (based upon testing by CS)
        zopflipng_options = " -y --lossy_transparent "
    saved_path = save_dir + "/val2014_png_" + s_quality + "/" + img.pre_filepath[-29:]
    if not os.path.isdir(save_dir + "/val2014_png_" + s_quality + "/"):
        os.makedirs(save_dir + "/val2014_png_" + s_quality + "/")
    shutil.copy(img.post_filepath,  saved_path)
    """
    zopflipng_command = (
        ZOPFLIPNG_EXE_PATH
        + zopflipng_options
        + shellquote(img.post_filepath)
        + " "
        + shellquote(img.post_filepath)
    )
    try:
        subprocess.check_output(zopflipng_command, stderr=subprocess.STDOUT, shell=True)
    except CalledProcessError as cpe:
        stdstream_lock.acquire()
        sys.stderr.write(
            ERROR_STRING
            + " "
            + img.pre_filepath
            + " processing failed at the zopflipng stage."
            + os.linesep
        )
        stdstream_lock.release()
        if is_gui(sys.argv):
            log_error(
                img.pre_filepath
                + " processing failed at the zopflipng stage. "
                + os.linesep
                + str(cpe)
            )
            return None
        else:
            raise cpe
    except Exception as e:
        if is_gui(sys.argv):
            log_error(
                img.pre_filepath
                + " processing failed at the pngquant stage. "
                + os.linesep
                + str(e)
            )
            return None
        else:
            raise e

    """
    # Check file size post-optimization and report comparison with pre-optimization file
    img.get_post_filesize()
    percent = img.get_compression_percent()
    percent_string = "{0:.2f}%".format(percent)
    # if compression occurred, color the percent string green
    # otherwise, leave it default text color
    if not is_gui(sys.argv) and percent < 100:
        percent_string = format_ansi_green(percent_string)

    # report percent original file size / post file path / size (bytes) to stdout (command line executable)
    stdstream_lock.acquire()
    print(
        "[" + str(inx) + "]" + "[ "
        + percent_string
        + " ] "
        + img.post_filepath[-36:]
        + " ("
        + str(img.post_size)
        + " bytes)"
    )
    pre = time.perf_counter()
    avg_post_file_size.append(img.post_size)
    avg_reduce_ratio.append(percent)
    stdstream_lock.release()

    # report percent original file size / post file path / size (bytes) to log file (macOS GUI + right-click service)
    if is_gui(sys.argv):
        log_info(
            "[ "
            + percent_string
            + " ] "
            + img.post_filepath
            + " ("
            + str(img.post_size)
            + " bytes)"
        )
    return saved_path

# -----------
# Utilities
# -----------


def fix_filepath_args(args):
    arg_list = []
    parsed_filepath = ""
    for arg in args:
        if arg[0] == "-":
            # add command line options
            arg_list.append(arg)
        elif len(arg) > 2 and "." in arg[1:]:
            # if format is `\w+\.\w+`, then this is a filename, not directory
            # this is the end of a filepath string that may have had
            # spaces in directories prior to this level.  Let's recreate
            # the entire original path
            filepath = parsed_filepath + arg
            arg_list.append(filepath)
            # reset the temp string that is used to reconstruct the filepaths
            parsed_filepath = ""
        else:
            # if the argument does not end with a .png, then there must have
            # been a space in the directory paths, let's add it back
            parsed_filepath = parsed_filepath + arg + " "
    # return new argument list with fixed filepaths to calling code
    return arg_list


def get_pngquant_path():
    if sys.argv[1] == "--gui":
        return "./pngquant"
    elif sys.argv[1] == "--service":
        return "/Applications/Crunch.app/Contents/Resources/pngquant"
    else:
        return PNGQUANT_CLI_PATH


def get_zopflipng_path():
    if sys.argv[1] == "--gui":
        return "./zopflipng"
    elif sys.argv[1] == "--service":
        return "/Applications/Crunch.app/Contents/Resources/zopflipng"
    else:
        return ZOPFLIPNG_CLI_PATH


def is_gui(arglist):
    return "--gui" in arglist or "--service" in arglist


def is_valid_png(filepath):
    # The PNG byte signature (https://www.w3.org/TR/PNG/#5PNG-file-signature)
    expected_signature = struct.pack("8B", 137, 80, 78, 71, 13, 10, 26, 10)
    # open the file and read first 8 bytes
    with open(filepath, "rb") as filer:
        signature = filer.read(8)
    # return boolean test result for first eight bytes == expected PNG byte signature
    return signature == expected_signature


def log_error(errmsg):
    current_time = time.strftime("%m-%d-%y %H:%M:%S")
    logging_lock.acquire()
    with open(LOGFILE_PATH, "a") as filewriter:
        filewriter.write(current_time + "\tERROR\t" + errmsg + os.linesep)
        filewriter.flush()
        os.fsync(filewriter.fileno())
    logging_lock.release()


def log_info(infomsg):
    current_time = time.strftime("%m-%d-%y %H:%M:%S")
    logging_lock.acquire()
    with open(LOGFILE_PATH, "a") as filewriter:
        filewriter.write(current_time + "\tINFO\t" + infomsg + os.linesep)
        filewriter.flush()
        os.fsync(filewriter.fileno())
    logging_lock.release()
    return None


def shellquote(filepath):
    return "'" + filepath.replace("'", "'\\''") + "'"


def format_ansi_red(text):
    if sys.stdout.isatty():
        return "\033[0;31m" + text + "\033[0m"
    else:
        return text


def format_ansi_green(text):
    if sys.stdout.isatty():
        return "\033[0;32m" + text + "\033[0m"
    else:
        return text


# ///////////////////////
# OBJECT DEFINITIONS
# ///////////////////////


class ImageFile(object):
    def __init__(self, filepath):
        self.pre_filepath = filepath
        self.post_filepath = self._get_post_filepath()
        self.pre_size = self._get_filesize(self.pre_filepath)
        self.post_size = 0

    def _get_filesize(self, file_path):
        return os.path.getsize(file_path)

    def _get_post_filepath(self):
        path, extension = os.path.splitext(self.pre_filepath)
        return path + "-crunch" + extension
    # traceback.format_exc()
    def get_post_filesize(self):
        self.post_size = self._get_filesize(self.post_filepath)

    def get_compression_percent(self):
        ratio = float(self.post_size) / float(self.pre_size)
        percent = ratio * 100
        return percent


def recv_msg(s):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(s, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(s, msglen)


def recvall(s, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = s.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def send_msg(c, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    c.sendall(msg)


def recv_helper(s, opt, all_path):
    max_inx = opt.number
    rate = opt.rate * 1024 / 8 # KB
    while True:
        try:
            start_counter = time.time()
            data = recv_msg(s)
            got_one = time.time()
            info = json.loads(str(data.decode('utf-8')))
            if info["code"] == 2 or info["code"] == 1:
                if info["code"] == 2:
                    # timestamp = info["timestamp"]
                    cpu_utilization.append(info["cpu"])
                    this_response = round(avg_processing_time[-1] + info["time"] + avg_size[-1]/rate, 3)
                    response.append(this_response)
                    compute_time.append(info["compute_time"])
                    # transmission_time.append(info["transmit"])
                    transmission_time.append(this_response - info["time"] - avg_processing_time[-1])
                    avg_complexity_yolo.append(avg_size[-1] * 1024 * 8)
                    if len(response) > 0:
                        print(
                            "+ %d %.2f response:%.2f(%s),[%.3f,:%.3f,%.3f],[%.2f KB], [cpu: %.2f]"
                            % (info["inx"],
                            got_one - start_counter,
                               this_response,
                               format_ansi_green("{0:.3f}s".format(np.average(response))),
                               np.average(avg_processing_time),
                               np.average(compute_time),
                               np.average(transmission_time),
                               np.average(avg_size),
                               np.average(cpu_utilization)
                               ))
                    if info["inx"] == max_inx - 1:
                        break
        except Exception as e:
            print(e.__str__())
            print("1 res=", list(np.array(response)))
            break
    print("all received")


# 28880
def send_helper(s, opt, all_path):
    global current_img_path_inx
    max_inx = opt.number
    try:
        current = time.time()
        if current_img_path_inx < max_inx:
            img_name = "p" + str(int(time.time())) + "_" + str(current_img_path_inx) + ".png"
            # print(current_img_path_inx, "compressing", all_path[current_img_path_inx])
            name = all_path[current_img_path_inx]
            if int(opt.max) < 100:
                saved_path = optimize_png(current_img_path_inx, opt.dir + "/" + name,
                                          str(opt.min + "-" + opt.max),
                                          str(opt.min + "_" + opt.max)
                                          , opt.speed, opt.save_dir)
            else:
                saved_path = opt.dir + "/" + name
            with open(saved_path, 'rb') as file:
                data = file.read()
            str_img = base64.encodebytes(data).decode("utf-8")
            # start_transmit = time.perf_counter()
            send_msg(s, json.dumps(
                {"app": opt.app, "code": 1, "max_inx": max_inx, "inx": current_img_path_inx,
                 "data": str_img, "name": img_name, "size": opt.size,
                 "timestamp": current, "start": time.time()}).encode(
                "utf-8"))
            # transmission_time.append(time.perf_counter() - start_transmit)
            file_size = os.path.getsize(saved_path) / 1024
            avg_size.append(file_size)
            current_img_path_inx += 1
        else:
            send_msg(s, json.dumps({"code": -1}).encode("utf-8"))
    except Exception as e:
        send_msg(s, json.dumps({"code": -1}).encode("utf-8"))
        s.close()
        print(e.__str__())
        print("2 res=", list(np.array(response)))
    # print("all sent")
    # print("res=", list(np.array(response[:300])))

import argparse
import socket
import json
import  base64
from _thread import *

#  3389: yolo 416
#  8080: pose
#  8001: yolo 608

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="/home/zxj/PyTorch-YOLOv3/data/coco/images/Crunch/src/images/val2014", help="dataset")
    parser.add_argument("--save_dir", type=str, default="/home/zxj/PyTorch-YOLOv3/data/coco/images/Crunch/src/images", help="dataset")
    parser.add_argument("--min", type=str, default="60", help="path to model definition file")
    parser.add_argument("--max", type=str, default="90", help="path to data config file")
    parser.add_argument("--speed", type=str, default="4", help="path to weights file")
    parser.add_argument("--port", type=int, default=3389, help="path to weights file")
    parser.add_argument("--host", type=str, default="192.168.1.163", help="path to weights file")
    parser.add_argument("--deadline", type=int, default=800, help="path to weights file")
    parser.add_argument("--type", type=str, default="png", help="path to weights file")
    parser.add_argument("--rate", type=int, default="10", help="path to weights file")
    parser.add_argument("--app", type=str, default="yolo", help="path to weights file")
    parser.add_argument("--number", type=int, default="310", help="path to weights file")
    parser.add_argument("--size", type=int, default="416", help="path to weights file")
    opt = parser.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((opt.host, opt.port))
    print("connected to ", opt.host)

    if opt.type == "jpg":
        opt.dir = opt.dir + "_old"
        with open('5k_filename_jpg.txt') as f:
            content = f.readlines()
    else:
        with open('5k_filename_png.txt') as f:
            content = f.readlines()

    # you may also want to remove whitespace characters like `\n` at the end of each line
    all_path = [x.strip() for x in content]

    if opt.deadline > 0:
        start_new_thread(recv_helper, (s, opt, all_path))
        # start_new_thread(send_helper, (s, opt, all_path))
        while current_img_path_inx < opt.number:
            timer = threading.Timer(opt.deadline / 1000, send_helper, [s, opt, all_path])
            timer.start()
            time.sleep(opt.deadline / 1000)
    else:
        current_img_path_inx = 0
        max_inx = 300
        saved_path = None
        avg_size = []
        avg_complexity_yolo = []
        try:
            while True:
                data = recv_msg(s)
                info = json.loads(str(data.decode('utf-8')))
                if info["code"] == 2 or info["code"] == 1:
                    if info["code"] == 2:
                        timestamp = info["timestamp"]
                        response.append(round(time.perf_counter() - timestamp - info["time"] + info["compute_time"], 3))
                        # transmission = time.perf_counter() - timestamp - info["time"]
                        # transmission_time.append(transmission)
                        avg_complexity_yolo.append(avg_size[-1] * 1024 * 8)
                        if len(transmission_time) > 0:
                            print(
                                "+ %s response:%s,compress:%.3f,compute:%.3f,transmit:%.3f,file:[%.2f] KB, rate: %.2f KB/s"
                                % (datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                                   format_ansi_green("{0:.3f}s".format(np.average(response))),
                                   np.average(avg_processing_time),
                                   np.average(info["compute_time"]),
                                   np.average(transmission_time),
                                   # np.average(avg_post_file_size) / 1024,
                                   np.average(avg_size),
                                   np.sum(avg_size) / np.sum(transmission_time)
                                   ))
                    if current_img_path_inx < max_inx:
                        img_name = "p" + str(int(time.time())) + "_" + str(current_img_path_inx) + ".png"
                        # print(current_img_path_inx, "compressing", all_path[current_img_path_inx])
                        name = all_path[current_img_path_inx]
                        current = time.perf_counter()
                        if int(opt.max) < 100:
                            saved_path = optimize_png(current_img_path_inx, opt.dir + "/" + name,
                                                  str(opt.min + "-" + opt.max),
                                                  str(opt.min + "_" + opt.max)
                                                  , opt.speed, opt.save_dir)
                        else:
                            saved_path = opt.dir + "/" + name
                        with open(saved_path, 'rb') as file:
                            data = file.read()
                        str_img = base64.encodebytes(data).decode("utf-8")
                        start_transmit = time.perf_counter()
                        send_msg(s, json.dumps(
                            {"app": opt.app, "code": 1, "max_inx": max_inx, "inx": current_img_path_inx,
                             "data": str_img, "name": img_name, "size": opt.size,
                             "timestamp": current}).encode(
                            "utf-8"))
                        transmission_time.append(time.perf_counter() - start_transmit)
                        file_size = os.path.getsize(saved_path) / 1024
                        avg_size.append(file_size)
                        current_img_path_inx += 1
                        if opt.deadline > 0:
                            time.sleep(1. * opt.deadline / 1000 - (time.perf_counter() - current))
                    else:
                        send_msg(s, json.dumps({"code": -1}).encode("utf-8"))
                        s.close()
                        break
        except Exception as e:
            send_msg(s, json.dumps({"code": -1}).encode("utf-8"))
            s.close()
            print(e.__str__())

    print("res=", list(np.array(response[:300])))

    #  50  + response:0.621s,compress:0.144,compute:0.208,transmit:0.028,file:[120.77] KB, rate: 4317.92 KB/s
    #  50  + response:0.310s,compress:0.143,compute:0.216,transmit:0.083,file:[120.77] KB, rate: 1454.80 KB/s

    # 100 + response:0.405s,compress:nan,compute:0.228,transmit:0.122,file:[156.87] KB, rate: 1288.44 KB/s



    #  computation_time = [0.16964, 0.16439, 0.16573, 0.16482, 0.16352, 0.15965, 0.13671, 0.0982, 0]
    #  file_size        = [104771, 112423, 124276, 138118, 155415, 171407, 182886, 186776, 466940]
    #  ratio            = [21.37, 23.15, 25.51, 28.36, 32, 35.64, 38.87, 40.3, 100]
    #  mAP              = [0.509, 0.511, 0.517, 0.521, 0.524, 0.526, 0.528, 0.530, 0.532]


    """
    100: jpg
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.298
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.532
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.304
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.152
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.325
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.424
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.264
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.390
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.234
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.428
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.553
    100: png
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.298
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.532
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.304
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.152
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.325
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.424
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.264
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.390
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.234
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.428
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.553
    20-30: png
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.286
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.509
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.290
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.134
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.310
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.415
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.257
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.377
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.392
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.209
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.547 
    30-40:png
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.287
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.511
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.293
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.135
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.311
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.418
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.257
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.379
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.393
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.214
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.548   
    40-50
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.290
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.517
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.293
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.315
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.423
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.258
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.381
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.396
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.218
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.419
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.550
    50-60
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.292
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.521
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.297
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.141
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.317
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.423
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.259
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.382
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.397
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.221
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.419
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551
    60-70
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.295
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.524
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.299
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.146
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.321
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.425
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.263
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.385
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.400
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.224
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.423
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.553
    70-80
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.295
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.526
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.299
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.145
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.321
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.426
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.262
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.386
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.401
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.225
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.423
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.553
    80-90
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.296
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.528
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.300
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.146
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.321
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.262
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.386
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.402
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.424
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.553
    """





    """
    VBoxManage bandwidthctl "t1" set Limit --limit 6m
    6m   3.38  0.541
    5m   2.93  0.581
    4m   2.63  0.619
    3m   2.18  0.696
    2m   1.62  0.842
    1m   0.83  1.389
    """

    """
     yolo416 = [1.304, 1.25, 1.046, 1.3, 1.504, 1.286, 1.235, 1.452, 1.204, 1.533, 1.398, 1.349, 1.194, 1.367, 1.476, 1.215, 1.089, 1.148, 1.444, 1.143, 1.091, 1.124, 1.061, 1.13, 1.016, 1.305, 1.219, 1.45, 1.114, 1.568, 1.221, 1.094, 1.286, 1.316, 1.094, 1.335, 1.002, 1.301, 1.179, 1.038, 1.297, 1.281, 1.344, 1.157, 1.166, 1.363, 1.202, 1.454, 1.218, 1.267, 1.178, 1.278, 1.313, 1.518, 1.351, 1.105, 1.295, 1.484, 1.091, 1.061, 1.151, 1.196, 1.46, 1.104, 1.126, 1.2, 1.34, 1.127, 1.337, 1.29, 1.13, 1.429, 1.247, 1.182, 1.331, 1.002, 1.405, 1.207, 1.074, 1.296, 1.204, 1.082, 1.412, 1.095, 1.048, 1.415, 1.361, 1.192, 1.108, 1.13, 1.109, 1.492, 1.116, 1.219, 1.134, 1.518, 1.219, 1.015, 1.219, 1.038, 1.451, 1.189, 1.187, 1.149, 1.219, 1.132, 1.637, 1.091, 1.137, 1.123, 1.587, 1.167, 0.987, 1.083, 1.566, 1.157, 1.224, 1.778, 1.656, 1.26, 1.237, 1.104, 1.71, 1.103, 1.036, 1.075, 1.661, 1.184, 1.498, 1.278, 1.309, 1.212, 1.26, 1.264, 1.136, 1.249, 1.674, 1.156, 1.361, 1.146, 1.641, 1.128, 1.084, 1.373, 1.587, 1.415, 1.244, 1.384, 1.225, 1.429, 1.186, 1.248, 1.149, 1.262, 1.344, 1.286, 1.276, 1.295, 1.497, 1.139, 1.15, 1.562, 1.108, 1.176, 1.377, 1.399, 1.213, 1.15, 1.157, 1.632, 1.342, 1.097, 1.358, 1.113, 1.515, 1.116, 1.051, 1.28, 1.388, 1.28, 1.435, 1.159, 1.143, 1.148, 1.262, 1.602, 1.284, 1.372, 1.255, 1.605, 1.366, 1.087, 1.158, 1.767, 1.082, 1.189, 1.123, 1.67, 1.104, 1.399, 1.283, 1.496, 1.22, 1.306, 1.181, 1.652, 1.095, 1.546, 1.291, 1.225, 1.591, 1.48, 1.344, 1.169, 1.267, 1.306, 1.216, 1.387, 1.226, 1.539, 1.126, 1.338, 1.112, 1.264, 1.293, 1.307, 1.102, 1.115, 1.087, 1.695, 1.196, 1.135, 1.121, 2.217, 1.894, 1.486, 1.326, 1.084, 1.145, 1.143, 1.375, 1.072, 1.467, 1.213, 1.356, 1.155, 1.197, 1.219, 1.349, 1.179, 1.026, 1.43, 1.168, 1.452, 1.222, 1.136, 1.138, 1.724, 1.33, 1.065, 1.628, 1.379, 1.397, 1.33, 1.127, 1.559, 1.365, 1.274, 1.537, 1.308, 1.152, 1.387, 1.118, 1.147, 1.377, 1.267, 1.263, 1.127, 1.418, 1.263, 1.261, 1.45, 1.08, 1.393, 1.231, 1.45, 1.092, 1.101, 1.347, 1.111, 1.122, 1.571, 1.305, 1.099, 1.436, 1.104, 1.213, 1.123, 1.427, 1.196]
     yolo416_avg = [1.224, 1.129, 0.978, 1.26, 1.267, 1.199, 1.175, 1.264, 1.186, 1.19, 1.08, 1.212, 1.212, 1.083, 1.35, 1.123, 1.112, 1.112, 1.15, 1.065, 1.126, 1.161, 1.103, 1.076, 1.015, 1.128, 1.125, 1.259, 1.155, 1.151, 1.104, 1.04, 1.263, 1.191, 1.115, 1.281, 1.031, 1.085, 1.126, 1.052, 1.148, 1.196, 1.105, 1.231, 1.168, 1.124, 1.183, 1.088, 1.057, 1.3, 1.277, 1.18, 1.162, 1.196, 1.055, 1.064, 1.266, 1.136, 1.128, 1.199, 1.11, 1.082, 1.132, 1.119, 1.121, 1.232, 1.24, 1.147, 1.22, 1.234, 1.15, 1.28, 1.215, 1.528, 1.327, 1.021, 1.217, 1.074, 1.096, 1.11, 1.146, 1.156, 1.1, 1.085, 1.082, 1.24, 1.155, 1.207, 1.106, 1.171, 1.188, 1.187, 1.092, 1.254, 1.174, 1.249, 1.204, 0.986, 1.204, 1.022, 1.285, 1.12, 1.169, 1.191, 1.125, 1.209, 1.011, 1.153, 1.176, 1.156, 1.212, 1.024, 1.095, 1.105, 1.336, 1.071, 1.26, 1.374, 1.095, 1.138, 1.194, 1.13, 1.202, 1.084, 0.995, 1.136, 1.162, 1.086, 1.352, 1.075, 1.123, 1.08, 1.036, 1.16, 1.109, 1.16, 1.299, 1.294, 1.321, 1.115, 1.142, 1.103, 1.014, 1.232, 1.106, 1.306, 1.195, 1.208, 1.088, 1.036, 1.048, 1.155, 1.192, 1.125, 1.053, 1.18, 1.295, 1.162, 1.38, 1.009, 1.166, 1.128, 1.054, 1.168, 1.102, 1.241, 1.092, 1.115, 1.385, 1.138, 1.4, 1.061, 1.182, 1.761, 1.421, 1.154, 1.012, 1.111, 1.159, 1.251, 1.249, 1.027, 1.192, 1.232, 1.215, 1.155, 1.3, 1.33, 1.104, 1.01, 1.22, 1.045, 1.079, 1.25, 1.079, 1.171, 1.116, 1.193, 1.148, 1.204, 1.117, 1.261, 1.229, 1.159, 1.207, 1.262, 1.047, 1.311, 0.98, 1.003, 1.33, 1.345, 1.088, 1.072, 1.141, 1.14, 1.224, 1.292, 1.22, 1.323, 1.047, 1.188, 1.073, 1.122, 1.047, 1.212, 1.098, 1.044, 1.144, 1.138, 1.144, 1.155, 1.0, 1.437, 1.092, 1.233, 1.244, 1.065, 1.128, 1.088, 1.234, 1.2, 1.11, 1.113, 1.342, 1.024, 1.123, 1.125, 1.104, 1.109, 1.094, 1.054, 1.153, 1.186, 1.238, 1.124, 1.036, 1.147, 1.164, 1.0, 1.37, 1.282, 1.088, 1.079, 1.203, 1.36, 1.282, 1.092, 1.286, 1.002, 1.084, 1.286, 1.16, 1.02, 1.202, 1.091, 1.207, 1.054, 1.18, 1.216, 1.129, 1.25, 1.07, 1.275, 1.125, 1.137, 1.033, 0.987, 1.125, 1.071, 1.127, 1.316, 1.15, 1.048, 1.147, 1.136, 1.061, 1.096, 1.302, 1.147]
         
    """