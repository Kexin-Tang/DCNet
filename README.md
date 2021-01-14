# Context-aware & Cascade-refinement Network (DCNet)

本文是DCNet食用手册，将会详细介绍如何使用DCNet，并介绍如何定制"个性化"DCNet :P

---

## 文件简介

<table>
	<tr>
		<td>路径</td>
		<td>文件名</td>
		<td>作用</td>
	</tr>
	<tr>
		<td>./data</td>
		<td>PIOD</td>
		<td>训练/测试数据集</td>
	</tr>
	<tr>
		<td rowspan="3">./detect_occ</td>
		<td>run.sh</td>
		<td>训练/测试脚本</td>
	</tr>
	<tr>
		<td>train.py</td>
		<td>训练模型</td>
	</tr>
	<tr>
		<td>test.py</td>
		<td>使用模型对图片进行处理得到结果</td>
	</tr>
	<tr>
		<td rowspan="2">./detect_occ/utils</td>
		<td>compute_loss.py</td>
		<td>Loss Function</td>
	</tr>
	<tr>
		<td>custom_transforms.py</td>
		<td>对输入图进行Transform</td>
	</tr>
	<tr>
		<td>./models</td>
		<td>网络名称.py</td>
		<td>对应网络的结构定义</td>
	</tr>
	<tr>
		<td>./lib/dateset</td>
		<td>occ_dataset.py</td>
		<td>PyTorch Dataset和Dataloader重写</td>
	</tr>
	<tr>
		<td>./experiments</td>
		<td>网络名称</td>
		<td>对应网络执行得到的结果图片</td>
	</tr>
	<tr>
		<td>./experiments/output</td>
		<td>网络名称</td>
		<td>对应网络训练得到的模型</td>
	</tr>
	<tr>
		<td rowspan="2">./experiments/configs</td>
		<td>train.yaml</td>
		<td>训练时的config</td>
	</tr>
	<tr>
		<td>test.yaml</td>
		<td>测试时的config</td>
	</tr>
</table>


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

# 训练方法

1. 更改`./experiments/configs/train.yaml`中对应的数据集path，网络arch名称，迭代次数，学习率等
2. 更改`./detect_occ/run.sh`中的注释行 
3. 在`./detect_occ`路径下运行
```bash
sh run.sh
```

---

# 测试方法

1. 更改`./experiments/configs/test.yaml`中对应的网络arch名称等
2. 更改`./detect_occ/run.sh`中的注释行
> 注：此处需要注意，在`test.py`中如果使用*myDataset*时，每次读入一张img，需要在`run.sh`中设置`--img`；如果使用*PIODDataset*则不需要使用`--img`。详细代码见`./lib/dataset/occ_dataset.py`
3. 在`./detect_occ`路径下运行
```bash
sh run.sh
```

---

## DIY方法

1. 需要修改`./lib/dataset/occ_dataset`中数据的路径，list_file及读取方式(即是否包括.type,包含`[:,-4]`与`+'.jpg'`两处)

2. 需要修改`./experiments/configs/train(test).yaml`中的`train_image_set`，`val_image_set`以`arch(网络的名字)`

3. 如果要使用新的pth，需要修改`./detect_occ/run.sh`中的路径

4. 如果在`'./models/'`中定义了新的network，需要同时更改`./models/__init__.py`
