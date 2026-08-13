from ultralytics.nn.tasks import yaml_model_load, parse_model
from ultralytics.utils.torch_utils import model_info  # 新增这行

cfg = yaml_model_load("./v8/my_yolov8n.yaml") 
model, save = parse_model(cfg, ch=3)
model_info(model, verbose=True)  # 替换原来的print(model.info())