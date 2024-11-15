from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import torch
import onnxruntime as ort
import numpy as np
import torch
import torch.onnx
import torch.nn as nn
from PIL import Image
import manager

manager = manager.Image_manager(imageorin)




def predict(imageoringin,size):
    """
    模型预测
    Args:
        imageoringin: 输入图像
        size: 输入尺寸
    Returns: 输出结果
    """
    ort_session = ort.InferenceSession("yolov8n.onnx")
    
    image = resize_image(imageoringin,size,1)
    
    # 获取输入输出的名称
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    input_data = np.array(image, dtype=np.float32) / 255.0
    input_data = input_data.transpose(2, 0, 1)  # 如果模型需要NCHW格式
    input_data = np.expand_dims(input_data, axis=0)  # 增加batch维度
    # 进行推理
    result = ort_session.run([output_name], {input_name: input_data})
    return result



def process_frame(imageoringin):
    size = [640,640]
    conf_thres = 0.6
    iou_thres = 0
    image = resize_image(imageoringin, size, 1)
    result = predict(image,size)
    result = std_output(result)
    result = nms(result, conf_thres, iou_thres)
    if result != []:
        ret = cod_trf(result, imageoringin, image)
        image = draw(ret,imageoringin,["people","2","3"])
    else:
        image = imageoringin
    return image

def Camera_open():
    # 获取摄像头,传入0表示获取系统默认摄像头
    cap = cv2.VideoCapture(1)

    # 打开cap
    cap.open(0)

    # 无限循环,直到break被触发
    while cap.isOpened():
        
        # 获取画面
        success, imageoringin = cap.read()
        
        if not success: # 如果获取画面不成功,则退出
            print('获取画面不成功,退出')
            break
        
        ## 逐帧处理
        image = process_frame(imageoringin)
        
        # 展示处理后的三通道图像
        cv2.imshow('my_window',image)
        
        key_pressed = cv2.waitKey(60) # 每隔多少毫秒毫秒,获取键盘哪个键被按下
        # print('键盘上被按下的键：', key_pressed)

        if key_pressed in [ord('q'),27]: # 按键盘上的q或esc退出(在英文输入法下)
            break
        
    # 关闭摄像头
    cap.release()
    # 关闭图像窗口
    cv2.destroyAllWindows()


def save_predict(path):
    # image_path = "two_runners1.jpg"
    image_path = path
    imageoringin = cv2.imread(image_path)
    image = process_frame(imageoringin)
    cv2.imwrite('output_image.jpg', image)


def run(model):
    if model == "camera":
        Camera_open()
    else:
        save_predict(model)
        

run("two_runners1.jpg")

