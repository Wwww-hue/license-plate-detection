# 项目介绍
本项目基于 Tkinter 和 OpenCV，采用 HOG 特征提取与 SVM 分类器的传统机器学习方法，构建了一个完整的车牌识别系统：支持从原始图像中识别车牌区域，分割并识别车牌字符，输出准确的车牌号码与颜色。
并集成了基础图像处理模块。

# 核心模块
项目可分为六个核心模块，依次为图像处理模块，模型训练模块，主程序gui模块，配置模块，数据资源模块，构建与发布模块。

## 图像处理模块
```
│   image_processing.py
```
此模块提供基础图像处理功能，包括：图像运算，图像增强，图像变换，图像滤波，形态学处理，边缘检测。

## 模型训练模块
```
│   predict.py
```
此模块：
定义 SVM 模型类与字符识别逻辑；

使用训练数据集（来自 train 文件夹）进行模型训练；

训练完成后保存为 svm.dat 和 svmchinese.dat；

提供预测函数对字符图像进行分类识别，识别结果如下：
![image](https://github.com/user-attachments/assets/0a21863c-d551-4d7c-9109-d261678cb9f0)
![image](https://github.com/user-attachments/assets/1162e663-9a73-453d-95d2-28bb4a3ad594)

## 主程序gui模块
```
│   surface.py
```
此模块使用 Tkinter 创建图形化界面，用于展示项目功能，如图所示：
![image](https://github.com/user-attachments/assets/43a18f6e-8181-4aa5-8d92-4719419e25a5)

## 配置模块
```
│   config.js
```
此模块存储图像处理和识别过程中的参数（如尺寸阈值、颜色筛选等），通过 JSON 格式便于修改和读取。

## 数据资源模块
```
│   svm.dat
│   svmchinese.dat
├───train
│   ├───chars2
│   └───charsChinese
├───test_image
```
train：字符图像训练数据集；

test_image：用于测试的车牌图像；

svm.dat / svmchinese.dat：训练得到的模型参数。

## 构建与发布模块
```
├───build
├───dist
```
将整个项目打包为桌面应用程序，无需 Python 环境即可运行。点击dist文件夹里的surface.exe即可运行。
如图：
![image](https://github.com/user-attachments/assets/70d8b5a7-25b2-446e-a227-9cbc0629f9b6)

# 部署方式










