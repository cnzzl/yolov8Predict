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

def resize_image(image, size, letterbox_image):
    """
        对输入图像进行resize
    Args:
        size:目标尺寸
        letterbox_image: bool 是否进行letterbox变换
    Returns:指定尺寸的图像
    """
    from PIL import Image
    ih, iw, _ = image.shape
    h, w = size
    if letterbox_image:
        scale = min(w/iw, h/ih)       # 缩放比例
        nw = int(iw*scale)
        nh = int(ih*scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # 生成画布
        image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
        # 将image放在画布中心区域-letterbox
        image_back[(h-nh)//2: (h-nh)//2 + nh, (w-nw)//2:(w-nw)//2+nw, :] = image
    else:
        image_back = image
    return image_back  

def std_output(pred):
    """
    将(1,84,8400)处理成(8400, 85)  85= box:4  conf:1 cls:80
    """
    pred = np.squeeze(pred) 
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    return pred  #(8400,85)

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

def nms(pred, conf_thres, iou_thres):
    """
    非极大值抑制nms
    Args:
        pred: 模型输出特征图
        conf_thres: 置信度阈值
        iou_thres: iou阈值
    Returns: 输出后的结果
    """
    box = pred[pred[..., 4] > conf_thres]  # 置信度筛选
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))

    total_cls = list(set(cls))  # 记录图像内共出现几种物体
    output_box = []
    # 每个预测类别分开考虑
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        temp = box[:, :6]
        for j in range(len(cls)):
            # 记录[x,y,w,h,conf(最大类别概率),class]值
            if cls[j] == clss:
                temp[j][5] = clss
                cls_box.append(temp[j][:6])
        #  cls_box 里面是[x,y,w,h,conf(最大类别概率),class]
        cls_box = np.array(cls_box)
        sort_cls_box = sorted(cls_box, key=lambda x: -x[4])  # 将cls_box按置信度从大到小排序
        # box_conf_sort = np.argsort(-box_conf)
        # 得到置信度最大的预测框
        max_conf_box = sort_cls_box[0]
        output_box.append(max_conf_box)
        sort_cls_box = np.delete(sort_cls_box, 0, 0)
        # 对除max_conf_box外其他的框进行非极大值抑制
        while len(sort_cls_box) > 0:
            # 得到当前最大的框
            max_conf_box = output_box[-1]
            del_index = []
            for j in range(len(sort_cls_box)):
                current_box = sort_cls_box[j]
                iou = get_iou(max_conf_box, current_box)
                if iou > iou_thres:
                    # 筛选出与当前最大框Iou大于阈值的框的索引
                    del_index.append(j)
            # 删除这些索引
            sort_cls_box = np.delete(sort_cls_box, del_index, 0)
            if len(sort_cls_box) > 0:
                output_box.append(sort_cls_box[0])
                sort_cls_box = np.delete(sort_cls_box, 0, 0)
    return output_box


def xywh2xyxy(*box):
    """
    将xywh转换为左上角点和左下角点
    Args:
        box:
    Returns: x1y1x2y2
    """
    ret = [box[0] - box[2] // 2, box[1] - box[3] // 2, \
          box[0] + box[2] // 2, box[1] + box[3] // 2]
    return ret

def get_inter(box1, box2):
    """
    计算相交部分面积
    Args:
        box1: 第一个框
        box2: 第二个框
    Returns: 相交部分的面积
    """
    x1, y1, x2, y2 = xywh2xyxy(*box1)
    x3, y3, x4, y4 = xywh2xyxy(*box2)
    # 验证是否存在交集
    if x1 >= x4 or x2 <= x3:
        return 0
    if y1 >= y4 or y2 <= y3:
        return 0
    # 将x1,x2,x3,x4排序,因为已经验证了两个框相交,所以x3-x2就是交集的宽
    x_list = sorted([x1, x2, x3, x4])
    x_inter = x_list[2] - x_list[1]
    # 将y1,y2,y3,y4排序,因为已经验证了两个框相交,所以y3-y2就是交集的宽
    y_list = sorted([y1, y2, y3, y4])
    y_inter = y_list[2] - y_list[1]
    # 计算交集的面积
    inter = x_inter * y_inter
    return inter

def get_iou(box1, box2):
    """
    计算交并比： (A n B)/(A + B - A n B)
    Args:
        box1: 第一个框
        box2: 第二个框
    Returns:  # 返回交并比的值
    """
    box1_area = box1[2] * box1[3]  # 计算第一个框的面积
    box2_area = box2[2] * box2[3]  # 计算第二个框的面积
    inter_area = get_inter(box1, box2)
    union = box1_area + box2_area - inter_area   #(A n B)/(A + B - A n B)
    iou = inter_area / union
    return iou

def cod_trf(result, pre, after):
    """
    因为预测框是在经过letterbox后的图像上做预测所以需要将预测框的坐标映射回原图像上
    Args:
        result:  [x,y,w,h,conf(最大类别概率),class]
        pre:    原尺寸图像
        after:  经过letterbox处理后的图像
    Returns: 坐标变换后的结果,并将xywh转换为左上角右下角坐标x1y1x2y2
    """
    res = np.array(result)
    x, y, w, h, conf, cls = res.transpose((1, 0))
    x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)  # 左上角点和右下角的点
    h_pre, w_pre, _ = pre.shape
    h_after, w_after, _ = after.shape
    scale = max(w_pre/w_after, h_pre/h_after)  # 缩放比例
    h_pre, w_pre = h_pre/scale, w_pre/scale  # 计算原图在等比例缩放后的尺寸
    x_move, y_move = abs(w_pre-w_after)//2, abs(h_pre-h_after)//2  # 计算平移的量
    ret_x1, ret_x2 = (x1 - x_move) * scale, (x2 - x_move) * scale
    ret_y1, ret_y2 = (y1 - y_move) * scale, (y2 - y_move) * scale
    ret = np.array([ret_x1, ret_y1, ret_x2, ret_y2, conf, cls]).transpose((1, 0))
    return ret  # x1y1x2y2

def draw(res, image, cls):
    """
    将预测框绘制在image上
    Args:
        res: 预测框数据
        image: 原图
        cls: 类别列表,类似["apple", "banana", "people"]  可以自己设计或者通过数据集的yaml文件获取
    Returns:
    """
    for r in res:
        # 画框
        image = cv2.rectangle(image, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255, 0, 0), 3)
        # 表明类别
        if r[5]<10:
        
            text = "{}:{}".format(cls[int(r[5])], \
                                round(float(r[4]), 2))
            h, w = int(r[3]) - int(r[1]), int(r[2]) - int(r[0])  # 计算预测框的长宽
            font_size = min(h/640, w/640) * 3  # 计算字体大小(随框大小调整)
            image = cv2.putText(image, text, (max(10, int(r[0])), max(20, int(r[1]))), cv2.FONT_HERSHEY_COMPLEX, max(font_size, 0.3), (0, 0, 255), 3)   # max()为了确保字体不过界
    # cv2.imshow("result", image)
    # cv2.waitKey()
    # cv2.destroyWindow("result")
    return image

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

