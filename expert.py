from ultralytics import YOLO

# 载入预训练模型
model = YOLO('yolov8n.pt')
# 保存
success = model.export(format='onnx')