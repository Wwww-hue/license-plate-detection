import cv2
import numpy as np


class ImageProcessor:
    @staticmethod
    def add_images(img1, img2):
        # 图像加法（逐像素相加），常用于图像融合
        return cv2.add(img1, img2)

    @staticmethod
    def subtract_images(img1, img2):
        # 图像减法，突出差异或进行背景去除
        return cv2.subtract(img1, img2)

    @staticmethod
    def logical_and(img1, img2):
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return cv2.bitwise_and(img1, img2)

    @staticmethod
    def logical_or(img1, img2):
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return cv2.bitwise_or(img1, img2)

    @staticmethod
    def adjust_brightness(img, value):
        # 调整亮度，将图像从BGR转为HSV，在V通道上加减亮度值
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(v.astype(np.int32) + value, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def adjust_contrast(img, value):
        # 调整对比度，通过alpha系数缩放像素值
        safe_value = max(value, 0.3)  # 防止对比度过低造成图像变灰
        return cv2.convertScaleAbs(img, alpha=safe_value)

    @staticmethod
    def equalize_histogram(img):
        # 直方图均衡化，提高图像的对比度
        if len(img.shape) == 2:  # 灰度图处理
            return cv2.equalizeHist(img)
        elif len(img.shape) == 3 and img.shape[2] == 3:  # 彩色图像处理（Y通道均衡）
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            raise ValueError("Unsupported image format")

    @staticmethod
    def rotate_image(img, angle):
        # 以图像中心为原点进行旋转
        height, width = img.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (width, height))

    @staticmethod
    def mirror_image(img, direction):
        # 镜像翻转，可水平、垂直或双向
        if direction == 'horizontal':
            return cv2.flip(img, 1)
        elif direction == 'vertical':
            return cv2.flip(img, 0)
        else:  # 同时水平+垂直
            return cv2.flip(img, -1)

    @staticmethod
    def mean_filter(img, kernel_size):
        # 均值滤波，对图像进行平滑处理
        return cv2.blur(img, (kernel_size, kernel_size))

    @staticmethod
    def median_filter(img, kernel_size):
        # 中值滤波，适合去除椒盐噪声
        return cv2.medianBlur(img, kernel_size)

    @staticmethod
    def gaussian_filter(img, kernel_size, sigma=0):
        # 高斯滤波，带权重的平滑处理
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    @staticmethod
    def erode(img, kernel_size):
        # 腐蚀操作，用于去除小的白色噪点或缩小前景
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(img, kernel)

    @staticmethod
    def dilate(img, kernel_size):
        # 膨胀操作，常用于加强前景或连接断裂部分
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(img, kernel)

    @staticmethod
    def opening(img, kernel_size):
        # 开运算：先腐蚀后膨胀，主要用于去除噪点
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def closing(img, kernel_size):
        # 闭运算：先膨胀后腐蚀，用于填补小的黑洞
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def canny_edge(img, threshold1, threshold2):
        # Canny边缘检测，常用于提取图像轮廓
        return cv2.Canny(img, threshold1, threshold2)

    @staticmethod
    def hough_lines(img):
        # 霍夫变换检测图像中的直线

        # 转为灰度图（若原图是彩色）
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 使用高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 使用Canny提取边缘
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        # 设置霍夫变换参数
        rho = 1  # 距离分辨率
        theta = np.pi / 2  # 角度分辨率
        threshold = 118  # 最小投票数阈值

        lines = cv2.HoughLines(edges, rho, theta, threshold)
        result = img.copy()

        # 如果检测到了直线
        if lines is not None:
            for i_line in lines:
                for rho, theta in i_line:
                    # 判断是垂直还是水平直线
                    if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):
                        # 垂直线
                        pt1 = (int(rho / np.cos(theta)), 0)
                        pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                        cv2.line(result, pt1, pt2, (0, 0, 255), 1)
                    else:
                        # 水平线
                        pt1 = (0, int(rho / np.sin(theta)))
                        pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                        cv2.line(result, pt1, pt2, (0, 0, 255), 1)

        return result

    @staticmethod
    def find_and_draw_contours(img):
        # 查找图像中的轮廓，并在图像上绘制

        result = img.copy()

        # 转为灰度图
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # 自适应阈值法进行二值化处理（抗光照变化能力强）
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            11, 3
        )

        # 提取外部轮廓
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 遍历每个轮廓
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < 100:  # 过滤掉小面积噪点
                continue

            # 绘制轮廓
            cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)

            # 计算并标记轮廓中心
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(result, (cX, cY), 3, (0, 255, 0), -1)
                cv2.putText(result, str(i + 1), (cX - 5, cY - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2, cv2.LINE_AA)

        return result
