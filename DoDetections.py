# encoding:utf-8

# import the necessary packages
import argparse
import os
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from imutils.object_detection import non_max_suppression
from models.research.object_detection.utils import label_map_util
from models.research.object_detection.utils import visualization_utils as vis_util

sys.path.append("../..")

start = time.process_time()

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('Cascades/haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier('Cascades/haarcascade_smile.xml')

# 下载下来的模型名
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
# 下载下来的模型的文件
PATH_TO_CKPT = 'models/research/object_detection/models/' + MODEL_NAME + '/frozen_inference_graph.pb'
# 数据集对应的label
PATH_TO_LABELS = os.path.join('models/research/object_detection/data', 'mscoco_label_map.pbtxt')
# 下载下来的模型的目录
model_path = "models/research/object_detection/models/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt"
# 数据集分类数量
NUM_CLASSES = 90

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, help="path to input image")
ap.add_argument("-east", "--east", type=str, default="EAST-TextDetector-Model/frozen_east_text_detection.pb",
                help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=640, help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=480, help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

end = time.process_time()
print("load the dependencies took {:.6f} seconds".format(end - start))


# 物体识别
def objectDetection(sess, detection_graph, category_index, frame):
    # 图片数据
    image_np = frame
    # 增加一个维度
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # 获取模型中的变量
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # 存放所有检测框
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # 每个检测结果的可信度
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # 每个框对应的类别
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # 检测框的个数
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # 开始计算
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded}
    )

    # 得到可视化结果
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=6
    )

    return image_np


# 人脸、眼睛、微笑识别
def faceDetection(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    # 识别人脸
    for (x, y, w, h) in faces:
        cv2.rectangle(
            image_np,
            (x, y),
            (x + w, y + h),
            (225, 0, 0),
            2
        )
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image_np[y:y + h, x:x + w]

        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=10,
            minSize=(5, 5)
        )
        # 识别眼睛
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 255, 0),
                2
            )

        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25)
        )

        # 识别微笑
        for (xx, yy, ww, hh) in smile:
            cv2.rectangle(
                roi_color,
                (xx, yy),
                (xx + ww, yy + hh),
                (0, 0, 255),
                2
            )

    return image_np


# 移动物体识别
def movingDetection(pre_frame, image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if pre_frame is None:
        pre_frame = gray
    else:
        img_delta = cv2.absdiff(pre_frame, gray)
        thresh = cv2.threshold(
            img_delta,
            25,
            255,
            cv2.THRESH_BINARY
        )[1]
        thresh = cv2.dilate(
            thresh,
            None,
            iterations=2
        )

        contours, hierarchy = cv2.findContours(
            thresh.copy(),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            if cv2.contourArea(c) < 1000:
                continue
            else:
                (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(
                image_np,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2
            )

            print("something is moving!!!")

    return pre_frame, image_np


# 计算文本区域检测的概率、文本区域的边界框坐标
def decode_predictions(scores, geometry):
    # 抓取score的维度，然后初始化两个列表
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for row in range(0, numRows):
        # 提取当前行的分数和几何数据
        scoresData = scores[0, 0, row]
        xData0 = geometry[0, 0, row]
        xData1 = geometry[0, 1, row]
        xData2 = geometry[0, 2, row]
        xData3 = geometry[0, 3, row]
        anglesData = geometry[0, 4, row]

        for col in range(0, numCols):
            # 忽略概率不高的区域来过滤弱文本检测
            if scoresData[col] < args["min_confidence"]:
                continue

            # 当图像通过网络时，EAST文本检测器自然地减少了体积大小，所以我们乘4使坐标回到原始图像
            (offsetX, offsetY) = (col * 4.0, row * 4.0)

            # 提取角度数据
            angle = anglesData[col]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # 计算文本区域的边框坐标
            height = xData0[col] + xData2[col]
            width = xData1[col] + xData3[col]
            endX = int(offsetX + cos * xData1[col] + sin * xData2[col])
            endY = int(offsetY - sin * xData1[col] + cos * xData2[col])
            startX = int(endX - width)
            startY = int(endY - height)

            # 更新rects和confidences数据库列表
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[col])

    return rects, confidences


# 文字识别
def textDetection(image):
    orig = image.copy()
    (H, W) = image.shape[:2]

    # 确定原始图像尺寸与新图像尺寸的比率（基于为--width和--height提供的命令行参数）
    (newW, newH) = (args["width"], args["height"])
    rW = W / float(newW)
    rH = H / float(newH)
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # 构建layerNames表
    # ● 第一层是我们的输出sigmoid激活，它给出了包含文本或不包含文本的区域的概率。
    # ● 第二层是表示图像“几何”的输出要素图。我们使用它来导出输入图像中文本的边界框坐标。
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ]

    print("loading EAST text detector...")

    # 使用cv2.dnn.readNet将神经网络加载到内存
    net = cv2.dnn.readNet(args["east"])

    # 从图像中构建一个blob以获得两个输出层集
    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (W, H),
        (123.68, 116.78, 103.94),
        swapRB=True,
        crop=False
    )

    tdstart = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    tdend = time.time()

    print("text detection took {:.6f} seconds".format(tdend - tdstart))

    # 计算文本区域检测的概率、文本区域的边界框坐标
    rects, confidences = decode_predictions(scores, geometry)

    boxes = np.array(rects)

    # 使用非最大值抑制，去除指向同一物体的重叠的边界框
    boxes = non_max_suppression(boxes, probs=confidences)

    # 循环遍历边界框
    for (startX, startY, endX, endY) in boxes:
        # 将坐标缩放到原始图像尺寸
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # 并将输出绘制到原始图像
        cv2.rectangle(
            orig,
            (startX, startY),
            (endX, endY),
            (0, 255, 0),
            2
        )

    return orig


# 移除视频数据的背景噪声
def removeBackgroundNoise(frame):
    # 利用Background Subtractor MOG2算法消除背景
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # 计算前景掩膜
    fgmask = fgbg.apply(frame, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)

    # 膨胀，使用特定的结构元素来侵蚀图像
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    # 使用掩膜移除静态背景
    res = cv2.bitwise_and(frame, frame, mask=fgmask)

    return res


# 视频数据的人体皮肤检测
def bodySkinDetection(frame):
    # 将移除背景后的图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 加高斯模糊
    blur = cv2.GaussianBlur(gray, (41, 41), 0)

    # 二值化处理
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if ret:
        return thresh


# 检测图像中的凸点(手指)个数
def getContours(frame):
    # 利用findContours检测图像中的轮廓, 其中返回值contours包含了图像中所有轮廓的坐标点
    contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    size = len(contours)
    maxArea = -1
    ci = 0
    if size > 0:
        for i in range(size):
            # 找到最大的轮廓（根据面积）
            temp = contours[i]
            # 计算轮廓区域面积
            area = cv2.contourArea(temp)
            length = cv2.arcLength(temp, True)
            if area > maxArea:
                maxArea = area
                ci = i

        # 得出最大的轮廓区域
        largestContour = contours[ci]

        return largestContour


# 计算欧氏距离
def getEucledianDistance(vector1, vector2):
    distant = vector1[0] - vector2[0]
    dist = np.sqrt(np.sum(np.square(distant)))
    return dist


# 计算有效手指个数
def getDefectsCount(largestContour, center, drawing, minDistance):
    fingerRes = []  # 寻找指尖
    maxDistance = 0
    count = 0
    notice = 0
    defectNumbers = 0
    for i in range(len(largestContour)):
        temp = largestContour[i]

        # 计算重心到轮廓边缘的距离
        dist = (temp[0][0] - center[0]) * (temp[0][0] - center[0]) + \
               (temp[0][1] - center[1]) * (temp[0][1] - center[1])
        if dist > maxDistance:
            maxDistance = dist
            notice = i
        if dist != maxDistance:
            count = count + 1
            if count > 40:
                count = 0
                maxDistance = 0
                flag = False  # 布尔值

                # # 低于手心的点不算
                # if center[1] < largestContour[notice][0][1]:
                #     continue

                # 离得太近的不算
                for j in range(len(fingerRes)):
                    if abs(largestContour[notice][0][0] - fingerRes[j][0]) < minDistance:
                        flag = True
                        break
                if flag:
                    continue
                fingerRes.append(largestContour[notice][0])

                # 画出指尖
                cv2.circle(
                    drawing,
                    tuple(largestContour[notice][0]),
                    8,
                    (255, 0, 0),
                    -1
                )
                cv2.line(
                    drawing,
                    center,
                    tuple(largestContour[notice][0]),
                    (255, 0, 0),
                    2
                )
                defectNumbers = defectNumbers + 1

    return defectNumbers, drawing


# 手势识别
def gestureDetection(array, minDistance, gesture):
    copy = array.copy()

    # 移除背景
    array = removeBackgroundNoise(array)
    thresh = bodySkinDetection(array)

    # 计算图像的轮廓
    largestContour = getContours(thresh.copy())

    if largestContour is not None:
        # 得出点集（组成轮廓的点）的凸包
        hull = cv2.convexHull(largestContour)

        # 画出最大区域轮廓
        cv2.drawContours(
            copy,
            [largestContour],
            0,
            (0, 255, 0),
            2
        )
        # 画出凸包轮廓
        cv2.drawContours(
            copy,
            [hull],
            0,
            (0, 0, 255),
            3
        )

        center = (0, 0)
        # 求最大区域轮廓的各阶矩
        moments = cv2.moments(largestContour)
        if moments['m00'] != 0:
            center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
        # 画出重心
        cv2.circle(
            copy,
            center,
            8,
            (0, 0, 255),
            -1
        )

        defectNumbers, drawing = getDefectsCount(largestContour, center, copy, minDistance)

        return drawing, defectNumbers
    else:
        return array, 0


# 调整最低阈值回调函数
def updateThresholdLow():
    x = cv2.getTrackbarPos('minDistance', 'DoDetections!')
    print(x)


# 调整最高阈值回调函数
def updateThresholdHigh():
    x = cv2.getTrackbarPos('gesture', 'DoDetections!')
    print(x)


def main():
    loadingStart = time.process_time()
    detection_graph = tf.Graph()

    # 设置默认的图
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
    # 将模型读取到默认的图中
    with tf.compat.v1.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    # 加载COCO数据标签，将 mscoco_label_map.pbtxt 的内容转换成
    # {1: {'id': 1, 'name': u'person'}...90: {'id': 90, 'name': u'toothbrush'}}格式
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=NUM_CLASSES,
        use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)

    loadingEnd = time.process_time()
    print("Loading models took {:.6f} seconds".format(loadingEnd - loadingStart))

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if cap is None:
        print('please connect the camera')
        exit()

    # 创建窗口
    cv2.namedWindow('DoDetections!', cv2.WINDOW_NORMAL)

    # 创建低阈值调节条
    cv2.createTrackbar('minDistance', 'DoDetections!', 0, 100, updateThresholdLow)
    # 创建高阈值调节条
    cv2.createTrackbar('gesture', 'DoDetections!', 1, 5, updateThresholdHigh)

    # 完成调节条的初始设置
    cv2.setTrackbarPos('minDistance', 'DoDetections!', 40)
    cv2.setTrackbarPos('gesture', 'DoDetections!', 1)

    # cv2.resizeWindow('DoDetections!', args["width"], args["height"] + 60)

    cap.set(3, args["width"])  # set Width
    cap.set(4, args["height"])  # set Height

    pre_frame = None

    while True:
        ret, img = cap.read()
        if not ret:
            break

        gesture = cv2.getTrackbarPos('gesture', 'DoDetections!')

        if gesture == 2:
            cv2.putText(
                img,
                "You chose the object detection!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )
            cv2.imshow('DoDetections!', img)

            # 物体识别
            with detection_graph.as_default():
                with tf.compat.v1.Session(graph=detection_graph) as sess:
                    # writer = tf.compat.v1.summary.FileWriter("logs/", sess.graph)
                    sess.run(tf.compat.v1.global_variables_initializer())

                    loader = tf.compat.v1.train.import_meta_graph(model_path + '.meta')
                    loader.restore(sess, model_path)
                    while True:
                        odstart = time.process_time()
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # press 'ESC' to quit
                        k = cv2.waitKey(30) & 0xff
                        if k == 27:
                            cv2.setTrackbarPos('gesture', 'DoDetections!', 1)
                            break

                        pre_gesture = cv2.getTrackbarPos('gesture', 'DoDetections!')
                        if pre_gesture != gesture:
                            cv2.setTrackbarPos('gesture', 'DoDetections!', pre_gesture)
                            break

                        image_np = objectDetection(sess, detection_graph, category_index, frame)

                        cv2.putText(
                            image_np,
                            "Doing object detection!",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 255),
                            2
                        )

                        odend = time.process_time()
                        print("One frame object detect took {:.6f} seconds".format(odend - odstart))
                        cv2.imshow('DoDetections!', image_np)
        elif gesture == 3:
            cv2.putText(
                img,
                "You chose the text detection!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )
            cv2.imshow('DoDetections!', img)

            # 文字识别
            while True:
                tdstart = time.process_time()
                ret, frame = cap.read()
                if not ret:
                    break

                # press 'ESC' to quit
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    cv2.setTrackbarPos('gesture', 'DoDetections!', 1)
                    break

                pre_gesture = cv2.getTrackbarPos('gesture', 'DoDetections!')
                if pre_gesture != gesture:
                    cv2.setTrackbarPos('gesture', 'DoDetections!', pre_gesture)
                    break

                image_np = textDetection(frame)
                cv2.putText(
                    image_np,
                    "Doing text detection!",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

                tdend = time.process_time()
                print("One frame text detect took {:.6f} seconds".format(tdend - tdstart))
                cv2.imshow('DoDetections!', image_np)
        elif gesture == 4:
            cv2.putText(
                img,
                "You chose the face detection!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )
            cv2.imshow('DoDetections!', img)

            # 人脸、眼睛、微笑识别
            while True:
                fdstart = time.process_time()
                ret, frame = cap.read()
                if not ret:
                    break

                # press 'ESC' to quit
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    cv2.setTrackbarPos('gesture', 'DoDetections!', 1)
                    break

                pre_gesture = cv2.getTrackbarPos('gesture', 'DoDetections!')
                if pre_gesture != gesture:
                    cv2.setTrackbarPos('gesture', 'DoDetections!', pre_gesture)
                    break

                image_np = faceDetection(frame)
                cv2.putText(
                    image_np,
                    "Doing face detection!",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

                fdend = time.process_time()
                print("One frame face detect took {:.6f} seconds".format(fdend - fdstart))
                cv2.imshow('DoDetections!', image_np)
        elif gesture == 5:
            cv2.putText(
                img,
                "You chose the moving detection!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )
            cv2.imshow('DoDetections!', img)

            # 移动物体识别
            while True:
                fdstart = time.process_time()
                ret, frame = cap.read()
                if not ret:
                    break

                # press 'ESC' to quit
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    cv2.setTrackbarPos('gesture', 'DoDetections!', 1)
                    break

                pre_gesture = cv2.getTrackbarPos('gesture', 'DoDetections!')
                if pre_gesture != gesture:
                    cv2.setTrackbarPos('gesture', 'DoDetections!', pre_gesture)
                    break

                pre_frame, image_np = movingDetection(pre_frame, frame)
                cv2.putText(
                    image_np,
                    "Doing moving detection!",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

                fdend = time.process_time()
                print("One frame face detect took {:.6f} seconds".format(fdend - fdstart))
                cv2.imshow('DoDetections!', image_np)
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # press 'ESC' to quit
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

                pre_gesture = cv2.getTrackbarPos('gesture', 'DoDetections!')
                if pre_gesture != gesture:
                    cv2.setTrackbarPos('gesture', 'DoDetections!', pre_gesture)
                    break

                # 获取阈值
                minDistance = cv2.getTrackbarPos('minDistance', 'DoDetections!')
                frame, gesture = gestureDetection(frame, minDistance, gesture)

                if (gesture == 2) | (gesture == 3) | (gesture == 4) | (gesture == 5):
                    cv2.setTrackbarPos('gesture', 'DoDetections!', gesture)

                cv2.putText(
                    frame,
                    "please choose a kind of detection",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

                cv2.imshow('DoDetections!', frame)
                break

        # press 'ESC' to quit
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
