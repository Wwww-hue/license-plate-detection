![image](https://github.com/user-attachments/assets/55153408-2478-4fc5-b2e9-833f370e5684)![image](https://github.com/user-attachments/assets/55153408-2478-4fc5-b2e9-833f370e5684)![image](https://github.com/user-attachments/assets/f1d56775-b56d-408f-b678-737aeaf1d7d1)![image](https://github.com/user-attachments/assets/f1d56775-b56d-408f-b678-737aeaf1d7d1)
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
此模块使用 Tkinter 创建图形化界面，用于展示项目功能，如图所示：
![image](https://github.com/user-attachments/assets/43a18f6e-8181-4aa5-8d92-4719419e25a5)








