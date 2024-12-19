import cv2
import numpy as np
class Image_manager:
    
    def __init__(self):
        self.imageorin = []
        self.image = []
        self.input_data = []
        self.result = []
        self.imageresult= []

    def reture_image(self):
        return self.image
#前处理
    def resize_image(self,image, size, letterbox_image):
        """
            对输入图像进行resize
        Args:
            size:目标尺寸
            letterbox_image: bool 是否进行letterbox变换
        Returns:指定尺寸的图像
        """
        self.image = image
        # self.imageorin = image
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
        input_data = np.array(image_back, dtype=np.float32) / 255.0
        input_data = input_data.transpose(2, 0, 1)  # 如果模型需要NCHW格式
        input_data = np.expand_dims(input_data, axis=0)  # 增加batch维度
        self.input_data = input_data
        self.image = image_back
##后处理
    def std_output(self):
        """
        将(1,84,8400)处理成(8400, 85)  85= box:4  conf:1 cls:80
        """
        result = self.result
        result = np.squeeze(result) 
        result = np.transpose(result, (1, 0))
        result_class = result[..., 4:]
        result_conf = np.max(result_class, axis=-1)
        result = np.insert(result, 4, result_conf, axis=-1)
        self.result = result  #(8400,85)

    def nms(self, conf_thres, iou_thres):
        """
        非极大值抑制nms
        Args:
            pred: 模型输出特征图
            conf_thres: 置信度阈值
            iou_thres: iou阈值
        Returns: 输出后的结果
        """
        result = self.result
        box = result[result[..., 4] > conf_thres]  # 置信度筛选
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
                    iou = self.get_iou(max_conf_box, current_box)
                    if iou > iou_thres:
                        # 筛选出与当前最大框Iou大于阈值的框的索引
                        del_index.append(j)
                # 删除这些索引
                sort_cls_box = np.delete(sort_cls_box, del_index, 0)
                if len(sort_cls_box) > 0:
                    output_box.append(sort_cls_box[0])
                    sort_cls_box = np.delete(sort_cls_box, 0, 0)
        self.result = output_box


    def xywh2xyxy(self,*box):
        """
        将xywh转换为左上角点和左下角点
        Args:
            box:
        Returns: x1y1x2y2
        """
        ret = [box[0] - box[2] // 2, box[1] - box[3] // 2, \
            box[0] + box[2] // 2, box[1] + box[3] // 2]
        return ret

    def get_inter(self,box1, box2):
        """
        计算相交部分面积
        Args:
            box1: 第一个框
            box2: 第二个框
        Returns: 相交部分的面积
        """
        x1, y1, x2, y2 = self.xywh2xyxy(*box1)
        x3, y3, x4, y4 = self.xywh2xyxy(*box2)
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

    def get_iou(self,box1, box2):
        """
        计算交并比： (A n B)/(A + B - A n B)
        Args:
            box1: 第一个框
            box2: 第二个框
        Returns:  # 返回交并比的值
        """
        box1_area = box1[2] * box1[3]  # 计算第一个框的面积
        box2_area = box2[2] * box2[3]  # 计算第二个框的面积
        inter_area = self.get_inter(box1, box2)
        union = box1_area + box2_area - inter_area   #(A n B)/(A + B - A n B)
        iou = inter_area / union
        return iou

    def cod_trf(self):
        """
        因为预测框是在经过letterbox后的图像上做预测所以需要将预测框的坐标映射回原图像上
        Args:
            result:  [x,y,w,h,conf(最大类别概率),class]
            pre:    原尺寸图像
            after:  经过letterbox处理后的图像
        Returns: 坐标变换后的结果,并将xywh转换为左上角右下角坐标x1y1x2y2
        """
        result = self.result
        pre = self.imageorin
        after = self.image

        res = np.array(result)
        x, y, w, h, conf, cls = res.transpose((1, 0))
        x1, y1, x2, y2 = self.xywh2xyxy(x, y, w, h)  # 左上角点和右下角的点
        h_pre, w_pre, _ = pre.shape
        h_after, w_after, _ = after.shape
        scale = max(w_pre/w_after, h_pre/h_after)  # 缩放比例
        h_pre, w_pre = h_pre/scale, w_pre/scale  # 计算原图在等比例缩放后的尺寸
        x_move, y_move = abs(w_pre-w_after)//2, abs(h_pre-h_after)//2  # 计算平移的量
        ret_x1, ret_x2 = (x1 - x_move) * scale, (x2 - x_move) * scale
        ret_y1, ret_y2 = (y1 - y_move) * scale, (y2 - y_move) * scale
        ret = np.array([ret_x1, ret_y1, ret_x2, ret_y2, conf, cls]).transpose((1, 0))
        self.result = ret  # x1y1x2y2

    def draw(self, cls):
        """
        将预测框绘制在image上
        Args:
            res: 预测框数据
            image: 原图
            cls: 类别列表,类似["apple", "banana", "people"]  可以自己设计或者通过数据集的yaml文件获取
        Returns:
        """
        res = self.result
        image = self.imageorin
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
        self.imageresult = image