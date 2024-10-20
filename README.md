# Main project README

**项目结构**

```bash
text_recognition_project/    # 项目根目录
│
├── assets/                  # 存放图片和其他静态资源
│   ├── docs/                # 用于项目文档中的图片
│   │   ├── overview.png     # 项目概述图片
│   │   ├── architecture.png # 项目架构图片
│   │   └── workflow.png     # 项目工作流程图片
│   └── logo.png             # 项目徽标（例如用于README.md）
│
├── configs/                 # 配置文件
│   ├── train_config.yaml    # 训练阶段的配置文件
│   ├── eval_config.yaml     # 评估和测试的配置文件
│   ├── infer_config.yaml    # 推理时的配置文件
│   └── model_config.yaml    # 模型架构和参数的配置文件
│
├── data/                    # 数据集及标签存放文件夹
│   ├── raw/                 # 原始数据
│   ├── labels/              # 数据标签
│   └── README.md            # 数据集说明文档
│
├── datasets/                # 数据处理相关代码
│   ├── data_loader.py       # 数据加载和数据集生成代码
│   ├── data_augmentation.py # 数据增强模块
│   ├── data_preprocess.py   # 数据预处理代码
│   └── dataset_utils.py     # 一些数据集相关的工具函数
│
├── docs/                    # 项目文档文件夹
│   ├── TRAINING.md          # 训练使用文档
│   ├── EVALUATION.md        # 模型评估使用文档
│   ├── INFERENCE.md         # 推理使用文档
│   └── README.md            # 项目总体说明文档
│
├── models/                  # 模型定义和存储
│   ├── classic_models/      # 存储经典模型
│   ├── custom_models/       # 存储自定义模型
│   └── utils.py             # 与模型相关的工具函数
│
├── modules/                 # 深度学习常用模块
│   ├── attention_module.py  # 注意力机制模块
│   ├── loss_functions.py    # 自定义损失函数
│   ├── optimizers.py        # 优化器模块
│   └── schedulers.py        # 学习率调度模块
│
├── scripts/                 # 脚本文件夹
│   ├── train.py             # 模型训练脚本
│   ├── evaluate.py          # 模型评估脚本
│   ├── inference.py         # 推理/预测脚本
│   └── fine_tune.py         # 模型微调脚本
│
├── tools/                   # 工具文件夹
│   ├── utils.py             # 一些常用工具函数
│   ├── metrics.py           # 评估指标计算
│   └── checkpoint_manager.py# 模型保存和加载工具
│
├── logs/                    # 训练和测试日志文件
│   └── tensorboard/         # TensorBoard 相关日志
│
├── saved_models/            # 训练好的模型权重和配置文件
│
├── requirements.txt         # 项目所需的依赖库列表
├── environment.yaml         # conda 环境文件（可选）
└── README.md                # 项目主说明文档

```
