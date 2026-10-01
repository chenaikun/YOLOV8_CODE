from ultralytics.nn.tasks import parse_model, yaml_model_load
from ultralytics.utils.torch_utils import model_info

# 加载带注意力的配置
cfg = yaml_model_load("./v8/my_yolov8n_cbam.yaml")
model, save = parse_model(cfg, ch=3)

# 打印模型结构
model_info(model, verbose=True)

# 验证第2层是不是C2f_CBAM
for i, layer in enumerate(model):
    if i == 2:
        print(f"\n第2层类型: {layer.__class__.__name__} ✅")
        print(f"第2层Bottleneck类型: {layer.m[0].__class__.__name__} ✅")

# class SEBlock(nn.Module):
#     def __init__(self,c1,reduction=10):
#         super().__init__()
#         self.pool=nn.AdaptiveAvgPool2d(1)
#         self.fc=nn.Sequential(
#             nn.Linear(c1,c1//reduction,bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(c1//reduction,c1,bias=True),
#             nn.Sigmoid()
#         )

#     def forward(self,x):
#         b,c,h,w=x.shape
#         #[B,C,H,W]->[B,C,1,1]->[B,C]
#         y=self.pool(x).view(b,c)
#         print(f"Squeeze后：{y.shape}")
#         #Excitation:[B,C]->[B,C]->[B,C,1,1]
#         y=self.fc(y).view(b,c,1,1)
#         print(f"Excitation后：{y.shape}")
#         #Scale:广播相乘
#         out=x*y.expand_as(x)
#         print(f"Scale后(最终输出):{out.shape}")
#         return out


# from ultralytics.nn.modules.block import Bottleneck
# import torch.nn as nn

# class BottleneckWithSE(nn.Module):
#     """在官方Bottleneck里加SE注意力"""
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), reduction=16):
#         super().__init__()
#         # 复用官方Bottleneck的卷积结构
#         self.cv1 = nn.Conv2d(c1, c2, k[0], padding=k[0]//2, bias=False)
#         self.bn1 = nn.BatchNorm2d(c2)
#         self.act1 = nn.SiLU()
#         self.cv2 = nn.Conv2d(c2, c2, k[1], padding=k[1]//2, bias=False)
#         self.bn2 = nn.BatchNorm2d(c2)
#         self.act2 = nn.SiLU()
#         self.se = SEBlock(c2, reduction)  # 加在这里！cv2之后、残差相加之前
#         self.shortcut = shortcut and c1 == c2

#     def forward(self, x):
#         # 标准Bottleneck前向
#         out = self.act1(self.bn1(self.cv1(x)))
#         out = self.bn2(self.cv2(out))
#         out = self.se(out)  # SE注意力加权
#         if self.shortcut:
#             out = out + x  # 残差连接
#         return self.act2(out)

# # 测试：和官方Bottleneck对比
# bottleneck_official = Bottleneck(c1=128, c2=128)
# bottleneck_se = BottleneckWithSE(c1=128, c2=128)

# dummy = torch.randn(2, 128, 40, 40)
# out1 = bottleneck_official(dummy)
# out2 = bottleneck_se(dummy)


# class ChannelAttention(nn.Module):
#     """通道注意力（类似SE，但用MaxPool+AvgPool双路）"""
#     def __init__(self, c, reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(c, c // reduction, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(c // reduction, c, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         return self.sigmoid(avg_out + max_out)  # [B,C,1,1]

# class SpatialAttention(nn.Module):
#     """空间注意力：在通道维度做池化，学空间权重"""
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # 通道维度：平均池化+最大池化 → [B,2,H,W]
#         avg_out = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
#         max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,H,W]
#         x_cat = torch.cat([avg_out, max_out], dim=1)  # [B,2,H,W]
#         print(f"  空间注意力输入: {x_cat.shape}")  # [2,2,40,40]
#         att = self.sigmoid(self.conv(x_cat))  # [B,1,H,W]
#         print(f"  空间注意力权重: {att.shape}")  # [2,1,40,40]
#         return x * att  # 广播到[B,C,H,W]

# class CBAM(nn.Module):
#     """CBAM = 通道注意力 + 空间注意力"""
#     def __init__(self, c, reduction=16, kernel_size=7):
#         super().__init__()
#         self.ca = ChannelAttention(c, reduction)
#         self.sa = SpatialAttention(kernel_size)

#     def forward(self, x):
#         x = x * self.ca(x)  # 通道加权
#         print(f"  通道注意力后形状: {x.shape}")  # 不变
#         x = self.sa(x)  # 空间加权
#         print(f"  空间注意力后形状: {x.shape}")  # 不变
#         return x

# # 测试
# cbam = CBAM(c=128)
# dummy = torch.randn(2, 128, 40, 40)
# output = cbam(dummy)
# print(f"\nCBAM最终输出: {output.shape} ✅")


# print(f"官方Bottleneck输出: {out1.shape}")
# print(f"加SE的Bottleneck输出: {out2.shape} ✅ 完全一致！")

# #测试
# se=SEBlock(c1=128,reduction=16)
# dumpy=torch.randn(2,128,40,40)
# output=se(dumpy)
# print(f"输入：{dumpy.shape}->输出：{output.shape} ✅形状不变!")

# cfg = yaml_model_load("./v8/my_yolov8n.yaml")
# model, save = parse_model(cfg, ch=3)

# # 遍历模型找C2f层，自动适配不同模块的输入通道获取逻辑
# for i, layer in enumerate(model):
#     if layer.__class__.__name__ == "C2f":
#         # 适配YOLOv8的Conv封装：cv1是Conv类，卷积权重在conv属性里
#         in_c = layer.cv1.conv.in_channels
#         dummy_input = torch.randn(2, in_c, 80, 80)
#         output = layer(dummy_input)
#         print(f"C2f层{i}: 输入{dummy_input.shape} → cv1输出{[2, layer.cv1.conv.out_channels, 80, 80]} → 最终输出{output.shape}")


# # 先缓存所有有输出通道的层的通道数
# layer_out_channels = {}
# for i, layer in enumerate(model):
#     cls_name = layer.__class__.__name__
#     if cls_name == "C2f":
#         layer_out_channels[i] = layer.cv2.conv.out_channels
#     elif cls_name == "Conv":
#         layer_out_channels[i] = layer.conv.out_channels
#     elif cls_name == "SPPF":
#         layer_out_channels[i] = layer.cv2.conv.out_channels
#     elif cls_name == "Detect":
#         layer_out_channels[i] = layer.nc

# # 专门计算Concat层的输出通道
# for i, layer in enumerate(model):
#     if layer.__class__.__name__ == "Concat":
#         # Concat的from字段存的是输入层索引
#         in_layers = layer.f if isinstance(layer.f, list) else [layer.f]
#         concat_out_c = 0
#         valid = True
#         for l_idx in in_layers:
#             if l_idx == -1:
#                 # -1代表上一层，取最后一个缓存的通道
#                 concat_out_c += list(layer_out_channels.values())[-1]
#             elif l_idx in layer_out_channels:
#                 concat_out_c += layer_out_channels[l_idx]
#             else:
#                 valid = False
#                 break
#         if valid:
#             print(f"第{i}层（Concat）输入来自层{in_layers}，输出通道: {concat_out_c}")


# # 先缓存所有层的输出通道
# layer_out_c = {}
# for i, layer in enumerate(model):
#     cls = layer.__class__.__name__
#     if cls == "C2f":
#         layer_out_c[i] = layer.cv2.conv.out_channels
#     elif cls == "Conv":
#         layer_out_c[i] = layer.conv.out_channels
#     elif cls == "SPPF":
#         layer_out_c[i] = layer.cv2.conv.out_channels
#     elif cls == "Detect":
#         layer_out_c[i] = layer.nc

# # 验证所有Concat层
# for i, layer in enumerate(model):
#     if layer.__class__.__name__ == "Concat":
#         in_layers = layer.f if isinstance(layer.f, list) else [layer.f]
#         calc_c = 0
#         for l in in_layers:
#             if l == -1:
#                 calc_c += list(layer_out_c.values())[-1]
#             else:
#                 calc_c += layer_out_c[l]
#         print(f"第{i}层Concat验证：计算值{calc_c} = 打印值{calc_c} ✅")
