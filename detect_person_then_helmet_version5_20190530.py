# coding: utf-8
import datetime

import numpy as np
import os
import time
import datetime
import base64
import sys
import tensorflow as tf
import cv2
import threading
import requests
from multiprocessing import Queue
from typing import Any

sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util

'''
author: birdguan
institution: SEU
version: 1.5.0
date: 2019-05-30
基本设置
@CAMERA_NUM:网络摄像头的数目
@camera_addressi:各个网络摄像头的RTSP取流地址
    --遵循：rtsp://用户名:密码@IP地址:RTSP端口（默认554）/流格式（一般Streaming）/Channels/通道号（一般主通道101）
@URL: 上传地址
@USERNAME: 上传所需用户名字段
@PASSWORD: 上传所需密码字段
@STATIONCODE: 地铁站站点编码
@STATIONNAME: 
@INERAL: second
@PERSON_FILEPATH: 存放人图像的文件路劲
@PERSON_WITH_OUT_HELMET_PATH: 未佩戴安全帽的工人所在场景的照片存放路径
'''

# ----START----用户自行设置参数区域----START----
CAMERA_NUM = 4
camera_address1 = "rtsp://admin:admin12345@192.168.0.22:554/Streaming/Channels/101"
camera_address2 = "rtsp://admin:admin12345@192.168.0.23:554/Streaming/Channels/101"
camera_address3 = "rtsp://admin:admin12345@192.168.0.27:554/Streaming/Channels/101"
camera_address4 = "rtsp://admin:admin12345@192.168.0.28:554/Streaming/Channels/101"

# camera_address1 = "rtsp://admin:futurexlab109@192.168.1.111:554/Streaming/Channels/101"
# camera_address2 = "rtsp://admin:futurexlab109@192.168.1.111:554/Streaming/Channels/101"
# camera_address3 = "rtsp://admin:futurexlab109@192.168.1.114:554/Streaming/Channels/101"
# camera_address4 = "rtsp://admin:futurexlab109@192.168.1.114:554/Streaming/Channels/101"
INTERAL = 180
PERSON_FILEPATH = "./person_image"
PERSON_WITH_OUT_HELMET_PATH = "./person without helmet"
URL = ""
URL2 = ""
USERNAME = r''
PASSWORD = r''
USERNAME2 = r''
PASSWORD2 = r''
STATIONCODE = ''
STATIONNAME = ''
# ----END------用户自行设置参数区域-----END----

# ----START----[无需]用户自行设置参数区域----START----
PATH_TO_PERSON_CKPT = 'personDetect/frozen_inference_graph.pb'
PATH_TO_PERSON_LABELS = 'personDetect/personDetect.pbtxt'
PATH_TO_HELMET_CKPT = 'helmetDetect/graph/20190530/frozen_inference_graph.pb'
PATH_TO_HELMET_LABELS = 'helmetDetect/data/helmetDetect.pbtxt'
PERSON_SCORE_THRESHOLD = 0.75
HELMET_SCORE_THRESHOLD = 0.4
NUM_HELMET_CLASSES = 2
NUM_PERSON_CLASSES = 1
width = 1280
height = 720
height_show = 720 // 2
width_show = 1280 // 2
# ----END------[无需]用户自行设置参数区域------END----

#
person_detection_graph = tf.Graph()
with person_detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_PERSON_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
helmet_detection_graph = tf.Graph()
with helmet_detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_HELMET_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

boxes_person_default = person_detection_graph.get_tensor_by_name('detection_boxes:0')
scores_person_default = person_detection_graph.get_tensor_by_name('detection_scores:0')
classes_person_default = person_detection_graph.get_tensor_by_name('detection_classes:0')
num_detections_person_default = person_detection_graph.get_tensor_by_name('num_detections:0')
img_tensor_person = person_detection_graph.get_tensor_by_name('image_tensor:0')

boxes_helmet_default = helmet_detection_graph.get_tensor_by_name('detection_boxes:0')
scores_helmet_default = helmet_detection_graph.get_tensor_by_name('detection_scores:0')
classes_helmet_default = helmet_detection_graph.get_tensor_by_name('detection_classes:0')
num_detections_helmet_default = helmet_detection_graph.get_tensor_by_name('num_detections:0')
img_tensor_helmet = helmet_detection_graph.get_tensor_by_name('image_tensor:0')

label_map = label_map_util.load_labelmap(PATH_TO_PERSON_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_PERSON_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 网络摄像头取流多线程
class readCaptureThread(threading.Thread):
    def __init__(self, thread_name, camera_id1, camera_id2, cap1, cap2, queue1, queue2, queue_status1, queue_status2):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.camera_id1 = camera_id1
        self.camera_id2 = camera_id2
        self.cap1 = cap1
        self.cap2 = cap2
        self.queue1 = queue1
        self.queue2 = queue2
        self.queue_status1 = queue_status1
        self.queue_status2 = queue_status2
        self.__running = threading.Event()

    def run(self):
        print("开始线程：", self.thread_name)
        send_frame_to_main_thread(self.camera_id1, self.camera_id2, self.cap1, self.cap2,
                                  self.queue1, self.queue2, self.queue_status1, self.queue_status2)
        print("退出线程：", self.thread_name)

    def stop(self):
        self.__running.clear()


def send_frame_to_main_thread(camera_id1, camera_id2, cap1, cap2, queue1, queue2, queue_status_1, queue_status_2):
    """
    子线程采集到的图像帧发送到主线程处理
    :param camera_id1:
    :param camera_id2:
    :param cap1:
    :param cap2:
    :param queue1:
    :param queue2:
    """
    print("CameraCap", str(camera_id1), ": ", cap1.isOpened())
    print("CameraCap", str(camera_id2), ": ", cap2.isOpened())
    queue1.put(cap1.isOpened(), 1)
    queue2.put(cap2.isOpened(), 1)
    cap1.read()
    cap2.read()
    video_frame = 0
    while True:
        video_frame += 1
        # 取流
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if video_frame == 25 * INTERAL:
            video_frame = 0
            print("摄像头", camera_id1, ret1)
            print("摄像头", camera_id2, ret2)
            if ret1:
                queue1.put(frame1, 2)
            if ret2:
                queue2.put(frame2, 2)
            queue_status_1.put(ret1, 2)
            queue_status_2.put(ret2, 2)


def person_detect(boxes_person, scores_person, classes_person, image):
    """
    返回检测到的人的图片
    :param boxes_person:
    :param scores_person:
    :param classes_person:
    :param image:
    :return: 人图片列表
    """
    person_images = []
    boxes_person_squeeze = np.squeeze(boxes_person)
    scores_person_squeeze = np.squeeze(scores_person)
    classes_person_squeeze = np.squeeze(classes_person)
    height_start_list = []
    height_end_list = []
    width_start_list = []
    width_end_list = []
    print("scores_person_squeeze.size: ", scores_person_squeeze.size)
    print("classes_person_squeeze.size: ", classes_person_squeeze.size)
    for i in range(scores_person_squeeze.size):
        if classes_person_squeeze[i] == 1 and scores_person_squeeze[i] > PERSON_SCORE_THRESHOLD:
            height_start = int((boxes_person_squeeze[i][0]) * height)
            width_start = int(boxes_person_squeeze[i][1] * width)
            height_end = int(boxes_person_squeeze[i][2] * height)
            width_end = int(boxes_person_squeeze[i][3] * width)
            height_start -= int(0.3 * (height_end - height_end))
            if height_start < 0:
                height_start = 0
            # 抠出的人像图片宽至少60pixel, 高至少120pixel才保存
            if width_end > width_start + 60 and height_end > height_start + 120:
                copy_image = image[height_start:height_end, width_start:width_end]
                cv2.imwrite(
                    PERSON_FILEPATH + "/" + str(datetime.datetime.now()).replace(".", "_").replace(":", "_") + ".jpg",
                    copy_image)
                person_images.append(copy_image)
                height_start_list.append(height_start)
                height_end_list.append(height_end)
                width_start_list.append(width_start)
                width_end_list.append(width_end)
    return person_images, height_start_list, height_end_list, width_start_list, width_end_list


def helmet_detect(boxes_helmet, scores_helmet, classes_helmet, person_image,
                  height_start, height_end,
                  width_start, width_end,
                  frame_current, location):
    """
    检测人图片中是否含有安全帽,将没有戴安全帽的人的图片保存到指定文件夹
    :param person_image:
    :param location: 地点名称string类型
    :param boxes_helmet:
    :param scores_helmet:
    :param classes_helmet:
    """
    boxes_helmet = np.squeeze(boxes_helmet)
    scores_helmet = np.squeeze(scores_helmet)
    classes_helmet = np.squeeze(classes_helmet)
    has_helmet = True
    for i in range(scores_helmet.size):
        if classes_helmet[i] == 2 and scores_helmet[i] > HELMET_SCORE_THRESHOLD:
            has_helmet = False
            break

    # 认定为没有带安全帽并保存到指定目录
    if not has_helmet:
        cv2.rectangle(frame_current, (int(width_start), int(height_start)), (int(width_end), int(height_end)),
                      (255, 0, 0), 2)
        capture_time = time.strftime("%Y-%m-%d %H:%M:%S")
        file_name = PERSON_WITH_OUT_HELMET_PATH + "/" + location + capture_time.replace(":", "_") + ".jpg"
        cv2.imwrite(file_name, frame_current)
        print("========>[ ATTENTION ]", file_name, "saved")
        if os.path.getsize(file_name) != 0:
            send_image(image_path=file_name, capture_time=capture_time)


def convert_image_to_base64(imagepath):
    """
    convert image to base64 format
    :param image:
    :return: base64
    """
    with open(imagepath, 'rb') as f:
        image_base64 = 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode()
        return image_base64


def send_base64_to_server(url, username, password, stationCode, captureTime, photoBase64):
    """
    send base64 format of photo to the server
    :param url:
    :param username:
    :param password:
    :param stationCode:
    :param captureTime:
    :param photoBase64:
    """
    data = {"userName": username, "passWord": password, "stationCode": stationCode, "captureTime": captureTime,
            "photoBase64": photoBase64}
    r = requests.post(url=url, json=data)
    print("========>", r.text)


def send_image(image_path, capture_time):
    """
    send image of person without helmet to ZHANSU Co.,Ltd.'s server
    :param capture_time:
    :param image_path:
    """
    base64 = convert_image_to_base64(image_path)
    send_base64_to_server(url=URL, username=USERNAME, password=PASSWORD, stationCode=STATIONCODE,
                          captureTime=capture_time,
                          photoBase64=base64)
    send_base64_to_server(url=URL2, username=USERNAME2, password=PASSWORD2, stationCode=STATIONCODE,
                          captureTime=capture_time,
                          photoBase64=base64)


def main():
    person_config = tf.ConfigProto()
    person_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    helmet_config = tf.ConfigProto()
    helmet_config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(graph=person_detection_graph, config=person_config) as person_sess:
        with tf.Session(graph=helmet_detection_graph, config=helmet_config) as helmet_sess:

            queue1 = Queue()
            queue2 = Queue()
            queue3 = Queue()
            queue4 = Queue()
            queue_status_1 = Queue()
            queue_status_2 = Queue()
            queue_status_3 = Queue()
            queue_status_4 = Queue()

            # 正式运行流
            cap1 = cv2.VideoCapture(camera_address1)
            cap2 = cv2.VideoCapture(camera_address2)
            cap3 = cv2.VideoCapture(camera_address3)
            cap4 = cv2.VideoCapture(camera_address4)

            # 测试流
            # cap1 = cv2.VideoCapture("testVideo5.mp4")
            # cap2 = cv2.VideoCapture("testVideo5.mp4")
            # cap3 = cv2.VideoCapture("testVideo5.mp4")
            # cap4 = cv2.VideoCapture("testVideo5.mp4")

            read_capture_thread1_2 = readCaptureThread("取流线程1", 1, 2, cap1, cap2,
                                                       queue1, queue2, queue_status_1, queue_status_2)
            read_capture_thread3_4 = readCaptureThread("取流线程2", 3, 4, cap3, cap4,
                                                       queue3, queue4, queue_status_3, queue_status_4)
            read_capture_thread1_2.start()
            read_capture_thread3_4.start()

            frame_pre1 = queue1.get(2)
            frame_pre2 = queue2.get(2)
            frame_pre3 = queue3.get(2)
            frame_pre4 = queue4.get(2)


            # if queue1.get(1) or queue2.get(1) or queue3.get(1) or queue4.get(1):
            while True:
                camera_status1 = queue_status_1.get(2)
                camera_status2 = queue_status_2.get(2)
                camera_status3 = queue_status_3.get(2)
                camera_status4 = queue_status_4.get(2)
                while (camera_status1 == False or camera_status2 == False
                       or camera_status3 == False or camera_status4 == False):
                    # 正式运行流
                    print("Attempt to connect...")
                    print("Camera 1 connection status: ", cap1.isOpened())
                    print("Camera 2 connection status: ", cap2.isOpened())
                    print("Camera 3 connection status: ", cap3.isOpened())
                    print("Camera 4 connection status: ", cap4.isOpened())
                    cap1 = cv2.VideoCapture(camera_address1)
                    cap2 = cv2.VideoCapture(camera_address2)
                    cap3 = cv2.VideoCapture(camera_address3)
                    cap4 = cv2.VideoCapture(camera_address4)

                    # 测试流
                    # cap1 = cv2.VideoCapture("testVideo5.mp4")
                    # cap2 = cv2.VideoCapture("testVideo5.mp4")
                    # cap3 = cv2.VideoCapture("testVideo5.mp4")
                    # cap4 = cv2.VideoCapture("testVideo5.mp4")

                print("=========> [time]  ", time.strftime("%Y-%m-%d %H:%M:%S"), "       [status]  normal <========")
                frame_current1 = queue1.get(2)
                frame_current2 = queue2.get(2)
                frame_current3 = queue3.get(2)
                frame_current4 = queue4.get(2)

                # 一号摄像头
                if not np.equal(frame_current1, frame_pre1).all():
                    print("处理一号摄像头")
                    img_expanded1 = np.expand_dims(frame_current1, axis=0)
                    (boxes_person, scores_person, classes_person, num_detections_person) = person_sess.run(
                        [boxes_person_default, scores_person_default, classes_person_default,
                         num_detections_person_default],
                        feed_dict={img_tensor_person: img_expanded1})
                    person_images, height_start_list, height_end_list, width_start_list, width_end_list = person_detect(
                        boxes_person,
                        scores_person,
                        classes_person,
                        frame_current1)
                    for index, person_image in enumerate(person_images):
                        person_image_expanded = np.expand_dims(person_image, axis=0)
                        (boxes_helmet, scores_helmet, classes_helmet, num_detections_helmet) = helmet_sess.run(
                            [boxes_helmet_default, scores_helmet_default, classes_helmet_default,
                             num_detections_helmet_default],
                            feed_dict={img_tensor_helmet: person_image_expanded})
                        helmet_detect(boxes_helmet, scores_helmet, classes_helmet, person_image,
                                      height_start_list[index], height_end_list[index], width_start_list[index],
                                      width_end_list[index], frame_current1,
                                      STATIONNAME + "1号摄像头")

                # 二号摄像头
                if not np.equal(frame_current2, frame_pre2).all():
                    print("处理二号摄像头")
                    img_expanded2 = np.expand_dims(frame_current2, axis=0)
                    (boxes_person, scores_person, classes_person, num_detections_person) = person_sess.run(
                        [boxes_person_default, scores_person_default, classes_person_default,
                         num_detections_person_default],
                        feed_dict={img_tensor_person: img_expanded2})
                    person_images, height_start_list, height_end_list, width_start_list, width_end_list = person_detect(
                        boxes_person, scores_person, classes_person, frame_current2)
                    for index, person_image in enumerate(person_images):
                        person_image_expanded = np.expand_dims(person_image, axis=0)
                        (boxes_helmet, scores_helmet, classes_helmet, num_detections_helmet) = helmet_sess.run(
                            [boxes_helmet_default, scores_helmet_default, classes_helmet_default,
                             num_detections_helmet_default],
                            feed_dict={img_tensor_helmet: person_image_expanded})
                        helmet_detect(boxes_helmet, scores_helmet, classes_helmet, person_image,
                                      height_start_list[index], height_end_list[index], width_start_list[index],
                                      width_end_list[index], frame_current2,
                                      STATIONNAME + "2号摄像头")

                # 三号摄像头
                if not np.equal(frame_current3, frame_pre3).all():
                    print("处理三号摄像头")
                    img_expanded3 = np.expand_dims(frame_current3, axis=0)
                    (boxes_person, scores_person, classes_person, num_detections_person) = person_sess.run(
                        [boxes_person_default, scores_person_default, classes_person_default,
                         num_detections_person_default],
                        feed_dict={img_tensor_person: img_expanded3})
                    person_images, height_start_list, height_end_list, width_start_list, width_end_list = person_detect(
                        boxes_person, scores_person, classes_person, frame_current3)
                    for index, person_image in enumerate(person_images):
                        person_image_expanded = np.expand_dims(person_image, axis=0)
                        (boxes_helmet, scores_helmet, classes_helmet, num_detections_helmet) = helmet_sess.run(
                            [boxes_helmet_default, scores_helmet_default, classes_helmet_default,
                             num_detections_helmet_default],
                            feed_dict={img_tensor_helmet: person_image_expanded})
                        helmet_detect(boxes_helmet, scores_helmet, classes_helmet, person_image,
                                      height_start_list[index], height_end_list[index], width_start_list[index],
                                      width_end_list[index], frame_current3,
                                      STATIONNAME + "3号摄像头")

                # 四号摄像头
                if not np.equal(frame_current4, frame_pre4).all():
                    print("处理四号摄像头")
                    img_expanded4 = np.expand_dims(frame_current4, axis=0)
                    (boxes_person, scores_person, classes_person, num_detections_person) = person_sess.run(
                        [boxes_person_default, scores_person_default, classes_person_default,
                         num_detections_person_default],
                        feed_dict={img_tensor_person: img_expanded4})
                    person_images, height_start_list, height_end_list, width_start_list, width_end_list = person_detect(
                        boxes_person, scores_person, classes_person, frame_current4)
                    for index, person_image in enumerate(person_images):
                        person_image_expanded = np.expand_dims(person_image, axis=0)
                        (boxes_helmet, scores_helmet, classes_helmet, num_detections_helmet) = helmet_sess.run(
                            [boxes_helmet_default, scores_helmet_default, classes_helmet_default,
                             num_detections_helmet_default],
                            feed_dict={img_tensor_helmet: person_image_expanded})
                        helmet_detect(boxes_helmet, scores_helmet, classes_helmet, person_image,
                                      height_start_list[index], height_end_list[index], width_start_list[index],
                                      width_end_list[index], frame_current4,
                                      STATIONNAME + "4号摄像头")


if __name__ == '__main__':
    main()
