# Context-aware & Cascade-refinement Network (DCNet)

本文是DCNet食用手册，将会详细介绍如何使用DCNet，并介绍如何定制"个性化"DCNet :P

---

## 文件简介

文件路径 | 文件功能
:---: | :---:
./data/fileList.py | 生成数据的list.txt，存储的为(filename).(filetype)
./data/mat2hdf5_edge_ori.py | 将PIOD数据集生成Augmentation
./detect_occ/run.sh | 运行脚本，内部设置有训练和测试两种模式
./detect_occ/train.py | 训练设置
./detect_occ/test.py | 测试设置
./detect_occ/utils/custom_transforms.py | 数据增强的操作
./detect_occ/utils/compute_loss.py | 计算loss的方法
./experiments/configs | 存储yaml文件，设置参数
./experiments/output | 存储训练后的模型
./lib/dataset/occ_dataset.py | 重载DataSet
./models | 定义模型结构

---

## 初始化方法

初始化方法参考了[DOOBNet](https://github.com/GuoxiaWang/DOOBNet)的方法

需要下载[PASCAL VOC 2010](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar)和[Annotations](https://drive.google.com/file/d/0B7DaWBKShuMBSkZ6Mm5RVmg5ck0/view?usp=sharing). 解压后将`JPEGImages`移动到PIOD文件夹。最终的文件夹目录如下:
```
PIOD
|_ Data
|  |_ <id-1>.mat
|  |_ ...
|  |_ <id-n>.mat
|_ JPEGImages 
|  |_ <id-1>.jpg
|  |_ ...
|  |_ <id-n>.jpg
|_ val_doc_2010.txt
```
然后调用`mat2hdf5_edge_ori.py`生成标注 
```
mkdir  PIOD/Augmentation

python mat2hdf5_edge_ori.py \
--dataset PIOD \
--label-dir PIOD/Data \
--img-dir PIOD/JPEGImages \
--piod-val-list-file PIOD/val_doc_2010.txt \
--output-dir PIOD/Augmentation
```

---

## DIY方法

1. 需要修改`./lib/dataset/occ_dataset`中数据的路径，list_file及读取方式(即是否包括.type,包含`[:,-4]`与`+'.jpg'`两处)

2. 需要修改`./experiments/configs/train(test).yaml`中的`train_image_set`，`val_image_set`以`arch(网络的名字)`

3. 如果要使用新的pth，需要修改`./detect_occ/run.sh`中的路径

4. 如果在`'./models/'`中定义了新的network，需要同时更改`./models/__init__.py`
