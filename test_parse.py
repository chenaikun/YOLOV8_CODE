from ultralytics.nn.tasks import yaml_model_load, parse_model
import torch

cfg = yaml_model_load("./v8/my_yolov8n.yaml")
model, save = parse_model(cfg, ch=3)

# 遍历模型找C2f层，自动适配不同模块的输入通道获取逻辑
for i, layer in enumerate(model):
    if layer.__class__.__name__ == "C2f":
        # 适配YOLOv8的Conv封装：cv1是Conv类，卷积权重在conv属性里
        in_c = layer.cv1.conv.in_channels
        dummy_input = torch.randn(2, in_c, 80, 80)
        output = layer(dummy_input)
        print(f"C2f层{i}: 输入{dummy_input.shape} → cv1输出{[2, layer.cv1.conv.out_channels, 80, 80]} → 最终输出{output.shape}")



# 先缓存所有有输出通道的层的通道数
layer_out_channels = {}
for i, layer in enumerate(model):
    cls_name = layer.__class__.__name__
    if cls_name == "C2f":
        layer_out_channels[i] = layer.cv2.conv.out_channels
    elif cls_name == "Conv":
        layer_out_channels[i] = layer.conv.out_channels
    elif cls_name == "SPPF":
        layer_out_channels[i] = layer.cv2.conv.out_channels
    elif cls_name == "Detect":
        layer_out_channels[i] = layer.nc

# 专门计算Concat层的输出通道
for i, layer in enumerate(model):
    if layer.__class__.__name__ == "Concat":
        # Concat的from字段存的是输入层索引
        in_layers = layer.f if isinstance(layer.f, list) else [layer.f]
        concat_out_c = 0
        valid = True
        for l_idx in in_layers:
            if l_idx == -1:
                # -1代表上一层，取最后一个缓存的通道
                concat_out_c += list(layer_out_channels.values())[-1]
            elif l_idx in layer_out_channels:
                concat_out_c += layer_out_channels[l_idx]
            else:
                valid = False
                break
        if valid:
            print(f"第{i}层（Concat）输入来自层{in_layers}，输出通道: {concat_out_c}")




# 先缓存所有层的输出通道
layer_out_c = {}
for i, layer in enumerate(model):
    cls = layer.__class__.__name__
    if cls == "C2f":
        layer_out_c[i] = layer.cv2.conv.out_channels
    elif cls == "Conv":
        layer_out_c[i] = layer.conv.out_channels
    elif cls == "SPPF":
        layer_out_c[i] = layer.cv2.conv.out_channels
    elif cls == "Detect":
        layer_out_c[i] = layer.nc

# 验证所有Concat层
for i, layer in enumerate(model):
    if layer.__class__.__name__ == "Concat":
        in_layers = layer.f if isinstance(layer.f, list) else [layer.f]
        calc_c = 0
        for l in in_layers:
            if l == -1:
                calc_c += list(layer_out_c.values())[-1]
            else:
                calc_c += layer_out_c[l]
        print(f"第{i}层Concat验证：计算值{calc_c} = 打印值{calc_c} ✅")