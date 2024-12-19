import cv2
import onnxruntime as ort
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import cv2


from predict_manager import Image_manager


mng = Image_manager()

def predict(img,size,mode):
    """
    模型预测
    Args:
        imageoringin: 输入图像
        size: 输入尺寸
    Returns: 输出结果
    """
    # mng.resize_image(img, size, 1)
    print(mng.image.shape)
    input_data = mng.input_data
    if mode == "onnx":
        ort_session = ort.InferenceSession("yolov8n.onnx")
        # 获取输入输出的名称
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        # image = resize_image(imageoringin,size,1)
        

        # image = mng.image
        
        # 进行推理
        result = ort_session.run([output_name], {input_name: input_data})
        return result
    elif mode == "trt":
        ##初始化engine
        f = open("your_model.engine", "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) #Logger.WARNING or Logger.ERROR
        engine = runtime.deserialize_cuda_engine(f.read())#导入序列化的engine
        context = engine.create_execution_context()#创建执行上下文

        # 准备输出数据
        output_data = np.empty(shape=(1,84,8400), dtype=np.float32)
        d_output = cuda.mem_alloc(output_data.nbytes)

        # 准备输出数据
        output_data = np.empty(shape=(1,84,8400), dtype=np.float32)
        d_output = cuda.mem_alloc(output_data.nbytes)

        # 创建cuda流
        stream = cuda.Stream(0)
        contiguous_input_data = np.ascontiguousarray(input_data, dtype=np.float32)

        d_input=cuda.mem_alloc(contiguous_input_data .nbytes)
        cuda.memcpy_htod(d_input, contiguous_input_data)


        # 运行推理
        context.execute_v2(bindings=[int(d_input), int(d_output)])

        # 将输出数据从输出缓冲区拷贝到CPU
        cuda.memcpy_dtoh(output_data, d_output)
        return output_data

   

 

def process_frame(imageoringin):
    size = [640,640]
    conf_thres = 0.6
    iou_thres = 0
    mng.imageorin = imageoringin
    mng.resize_image(imageoringin, size, 1)
    #推理
    mng.result = predict(imageoringin,size,"trt")
    mng.std_output()
    mng.nms(conf_thres, iou_thres)
    if mng.result != []:
        mng.cod_trf()
        mng.draw(["people","2","3"])
        image = mng.imageresult
    else:
        image = mng.imageorin
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
        

# run("two_runners1.jpg")
run("camera")
