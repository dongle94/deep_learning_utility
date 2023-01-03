import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import time
import os
import GPUtil
from PIL import Image
from threading import Thread
import multiprocessing as mp

DETECTION_MODEL_PATH = 'detect-smarteye-person-1.pb'
IMG_PATH1 = "000000.jpg"
IMG_PATH2 = "000001.jpg"

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()

    def run(self):
        while not self.stopped:
            GPUtil.showUtilization(all=True)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


def main():
    monitor = Monitor(5)
    detection_graph = tf1.Graph()

    allow_growth = True
    memory_fraction = 0

    tf_config = tf1.ConfigProto()
    tf_config.gpu_options.allow_growth = allow_growth
    tf_config.gpu_options.visible_device_list = '0'
    if memory_fraction and memory_fraction != 0:
        tf_config.gpu_options.per_process_gpu_memory_fraction = memory_fraction        # 24GB

    sess = tf1.Session(config=tf_config, graph=detection_graph)
    print(f"[{os.getpid()}]========== [GPU설정] allwo_growth: {allow_growth} / mem_fraction: {memory_fraction} ==========")
    print(f"[{os.getpid()}]========== 모델 로드 시작 ==========")
    with sess.as_default():
        with detection_graph.as_default():
            with tf1.gfile.GFile(DETECTION_MODEL_PATH, 'rb') as fid:
                od_graph_def = tf1.GraphDef()
                od_graph_def.ParseFromString(fid.read())
                tf1.import_graph_def(od_graph_def, name='')

            sess.graph.finalize()

        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        print(image_tensor)
    print(f"[{os.getpid()}]========== 모델 로드 완료 ==========")

    # Image Ready
    img = np.asarray(Image.open(IMG_PATH1).convert('RGB'))
    img_np = np.expand_dims(img, axis=0)
    time.sleep(10)

    # Inference
    print(f"[{os.getpid()}]========== 이미지 추론 시작 ==========")
    t = time.time()
    with sess.as_default() as sess:
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: img_np})
    et = time.time() - t
    print(f"[{os.getpid()}]========== 이미지 웜 업(1회) 시간: {et:.4f} ==========")

    # Image Ready
    img = np.asarray(Image.open(IMG_PATH2).convert('RGB'))
    img_np = np.expand_dims(img, axis=0)

    # Inference
    t = time.time()
    TEST_NUMBER = 100
    with sess.as_default() as sess:
        # Actual detection.
        for _ in range(TEST_NUMBER):
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: img_np})
    et = time.time() - t
    print(f"[{os.getpid()}]========== 이미지 {TEST_NUMBER}회 추론 완료: {et:.4f} sec ==========")

    monitor.stop()

if __name__ == '__main__':
    # 일반테스트
    main()

    # Multithreading Test
    #for _ in range(8):
    #    Thread(target=main, args=()).start()

    # Multiprocessing Test
    # for _ in range(3):
    #     ctx = mp.get_context('fork')
    #     ctx = mp.get_context('spawn')
    #     ctx.Process(target=main, args=()).start()

