# Context-aware & Cascade-refinement Network (DCNet)

本文是DCNet食用手册，将会详细介绍如何使用DCNet，并介绍如何定制"个性化"DCNet :P

## 文件简介

文件路径 | 文件功能
:---: | :---:
./data/fileList.py | 生成数据的list.txt，存储的为(filename).(filetype)
./detect_occ/run.sh | 运行脚本，内部设置有训练和测试两种模式
./detect_occ/train.py | 训练设置
./detect_occ/eval.py | 测试设置
./detect_occ/utils/custom_transforms.py | 数据增强的操作


## DIY方法

1. 需要修改`./lib/dataset/occ_dataset`中数据的路径，list_file及读取方式(即是否包括.type,包含`[:,-4]`与`+'.jpg'`两处)

2. 需要修改`./experiments/configs/train(test).yaml`中的`train_image_set`，`val_image_set`以`arch(网络的名字)`

3. 如果要使用新的pth，需要修改`./detect_occ/run.sh`中的路径

4. 如果在`'./models/'`中定义了新的network，需要同时更改`./models/__init__.py`
