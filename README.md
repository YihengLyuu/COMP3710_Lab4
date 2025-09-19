# demo2-recognition
COMP3710 Part 4 Recognition Tasks

## 项目简介
本项目实现了 **UNet** 模型在 **OASIS 脑部 MRI 数据集** 上的训练与推理，用于脑部组织分割任务。

包含两个主要脚本：
1. **训练脚本 (`4.train_unet_oasis.py`)**：完成数据加载、训练、验证、测试，并保存模型。
2. **推理脚本 (`4.infer_unet_oasis.py`)**：使用训练好的模型对新图像进行分割推理并输出可视化结果。

---

## 文件结构
demo2/
├── 4.train_unet_oasis.py # 训练脚本
├── 4.infer_unet_oasis.py # 推理脚本
├── runs_unet/ # 训练输出目录 (best.pth, final.pth)
├── oasis_unet_out/ # 推理输出目录
└── OASIS/ # 数据集
├── keras_png_slices_train/ # 训练集图像
├── keras_png_slices_seg_train/ # 训练集标注
├── keras_png_slices_validate/ # 验证集图像
├── keras_png_slices_seg_validate/ # 验证集标注
├── keras_png_slices_test/ # 测试集图像
└── keras_png_slices_seg_test/ # 测试集标注
