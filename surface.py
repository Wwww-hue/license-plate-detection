import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
from tkinter import *
import predict
import cv2
from PIL import Image, ImageTk
import time
from image_processing import ImageProcessor
import os


class Surface(ttk.Frame):
    pic_path = ""
    viewhigh = 600
    viewwide = 600
    update_time = 0
    thread = None
    thread_run = False
    # camera = None
    color_transform = {
        "green": ("绿牌", "#55FF55"),
        "yellow": ("黄牌", "#FFFF00"),
        "blue": ("蓝牌", "#6666FF")}

    def __init__(self, win):
        ttk.Frame.__init__(self, win)
        win.title("车牌识别")
        win.geometry('900x600')

        # 左右区域框架
        frame_left = ttk.Frame(self, width=600)
        frame_left.pack(side=LEFT, fill=BOTH, expand=True)

        # 创建左侧 Canvas
        self.left_canvas = tk.Canvas(frame_left)
        left_scrollbar_y = ttk.Scrollbar(frame_left, orient="vertical", command=self.left_canvas.yview)
        left_scrollbar_x = ttk.Scrollbar(frame_left, orient="horizontal", command=self.left_canvas.xview)
        self.left_canvas.configure(yscrollcommand=left_scrollbar_y.set, xscrollcommand=left_scrollbar_x.set)

        left_scrollable_frame = ttk.Frame(self.left_canvas)
        left_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        )
        self.left_canvas.create_window((0, 0), window=left_scrollable_frame, anchor="nw")

        ttk.Label(left_scrollable_frame, text='原图：').pack(anchor="nw", padx=5, pady=5)
        self.image_ctl = ttk.Label(left_scrollable_frame)
        self.image_ctl.pack(anchor="nw", padx=5, pady=5)

        left_scrollbar_y.pack(side="right", fill="y")
        left_scrollbar_x.pack(side="bottom", fill="x")
        self.left_canvas.pack(side="left", fill="both", expand=True)

        # 创建右侧 Canvas + 滚动区域
        self.right_canvas = tk.Canvas(self, width=300)
        right_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.right_canvas.yview)
        self.right_canvas.configure(yscrollcommand=right_scrollbar.set)

        right_scrollable_frame = ttk.Frame(self.right_canvas)
        right_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))
        )
        self.right_canvas.create_window((0, 0), window=right_scrollable_frame, anchor="nw")
        self.right_canvas.pack(side=RIGHT, fill=BOTH, expand=False)
        right_scrollbar.pack(side=RIGHT, fill=Y)

        # 创建右侧主内容框
        frame_right1 = ttk.Frame(right_scrollable_frame)
        frame_right1.pack(fill=tk.BOTH, expand=True)

        # 图像状态
        self.current_image = None
        self.original_image = None
        self.image_processor = ImageProcessor()

        # 右侧识别信息区
        ttk.Label(frame_right1, text='车牌位置：').pack(anchor="w", padx=5, pady=2)
        self.roi_ctl = ttk.Label(frame_right1)
        self.roi_ctl.pack(anchor="w", padx=5)

        ttk.Label(frame_right1, text='识别结果：').pack(anchor="w", padx=5, pady=2)
        self.r_ctl = ttk.Label(frame_right1, text="")
        self.r_ctl.pack(anchor="w", padx=5)

        self.color_ctl = ttk.Label(frame_right1, text="")
        self.color_ctl.pack(anchor="w", padx=5, pady=5)

        from_pic_ctl = ttk.Button(frame_right1, text="来自图片", width=25, command=self.from_pic)
        from_pic_ctl.pack(anchor="center", pady=10)
        reset_pic_ctl = ttk.Button(frame_right1, text="恢复原图", width=25, command=self.reset_image)
        reset_pic_ctl.pack(anchor="center", pady=5)

        # 初始化预测器
        self.predictor = predict.CardPredictor()
        self.predictor.train_svm()

        # 鼠标滚轮绑定：分别感知左右区域
        def _on_mousewheel_left(event):
            self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_right(event):
            self.right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_mousewheel(canvas_widget, handler):
            canvas_widget.bind("<Enter>", lambda e: canvas_widget.bind_all("<MouseWheel>", handler))
            canvas_widget.bind("<Leave>", lambda e: canvas_widget.unbind_all("<MouseWheel>"))

        bind_mousewheel(self.left_canvas, _on_mousewheel_left)
        bind_mousewheel(self.right_canvas, _on_mousewheel_right)

        # 添加图像处理控件（继续调用原方法）
        self.add_image_processing_controls(frame_right1)

    def get_imgtk(self, img_bgr):
        """
        将OpenCV的BGR格式图像转换为适合Tkinter显示的ImageTk格式，并根据视图大小自动缩放。

        参数：
        img_bgr: 输入的OpenCV图像，BGR格式的numpy数组。

        返回值：
        imgtk: 适合Tkinter控件显示的ImageTk.PhotoImage对象。
        """
        # OpenCV默认是BGR格式，转换成RGB格式，PIL才能正确识别颜色
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # 将numpy数组转换为PIL的Image对象
        im = Image.fromarray(img)
        # 将PIL Image转换为Tkinter兼容的PhotoImage对象
        imgtk = ImageTk.PhotoImage(image=im)

        # 获取当前图片宽度和高度（像素）
        wide = imgtk.width()
        high = imgtk.height()

        # 如果图片尺寸超过预设视图范围，则进行等比例缩放
        if wide > self.viewwide or high > self.viewhigh:
            # 计算宽度缩放比例
            wide_factor = self.viewwide / wide
            # 计算高度缩放比例
            high_factor = self.viewhigh / high
            # 取缩放比例的较小值，保证图片完全适应视图框且不变形
            factor = min(wide_factor, high_factor)

            # 根据缩放比例调整宽高（向下取整）
            wide = int(wide * factor)
            if wide <= 0:  # 防止宽度小于等于0
                wide = 1
            high = int(high * factor)
            if high <= 0:  # 防止高度小于等于0
                high = 1

            # 使用PIL的高质量重采样算法调整图像大小
            im = im.resize((wide, high), Image.Resampling.LANCZOS)
            # 重新转换为Tkinter的PhotoImage对象
            imgtk = ImageTk.PhotoImage(image=im)

        # 返回最终可直接用于Tkinter显示的图像对象
        return imgtk

    def show_roi(self, r, roi, color):
        """
        显示车牌区域（ROI）图像及相关信息。

        参数：
        r: 车牌识别结果文本（字符串），为空则表示无结果
        roi: 车牌区域图像，BGR格式的numpy数组
        color: 车牌颜色字符串（如"green"、"yellow"、"blue"）

        - 如果有识别结果r，将ROI图像转换为Tkinter显示格式并更新控件；
        - 同时显示识别文本和对应颜色标签（背景色）；
        - 如果无识别结果，则清空相关显示控件，禁用对应状态。
        """
        if r:
            # 转换颜色空间并转成Tkinter支持的格式
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)

            # 更新显示ROI图片的控件，启用显示状态
            self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            # 显示识别结果文本
            self.r_ctl.configure(text=str(r))
            # 更新时间戳
            self.update_time = time.time()

            try:
                # 根据颜色字典配置显示文字及背景色
                c = self.color_transform[color]
                self.color_ctl.configure(text=c[0], background=c[1], state='enable')
            except:
                # 颜色不存在则禁用颜色显示控件
                self.color_ctl.configure(state='disabled')
        else:
            # 无识别结果时，清空图片和文本，禁用相关控件
            self.roi_ctl.configure(image='', state='disabled')
            self.imgtk_roi = None
            self.r_ctl.configure(text="")
            self.color_ctl.configure(text="", background='SystemButtonFace', state='disabled')

    def from_pic(self):
        """
        从本地选择一张图片进行车牌识别处理。

        - 停止可能正在运行的线程
        - 设定默认打开目录为脚本目录下的test_image文件夹（存在则用它，否则用脚本目录）
        - 弹出文件选择对话框，选择jpg图片
        - 读取选中的图片并保存为当前图像和原始图像副本
        - 将图片转换为Tkinter显示格式并更新界面左侧显示
        - 调用预测器进行车牌识别，获得识别结果、车牌区域ROI及颜色
        - 调用show_roi函数显示识别结果和车牌区域
        """
        self.thread_run = False  # 停止线程标志

        base_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
        test_img_dir = os.path.join(base_dir, "test_image")  # 构造test_image目录路径

        # 弹出文件选择框，初始目录为test_image（存在则用它）
        self.pic_path = askopenfilename(
            title="选择识别图片",
            filetypes=[("jpg图片", "*.jpg")],
            initialdir=test_img_dir if os.path.exists(test_img_dir) else base_dir
        )

        if self.pic_path:
            # 使用predict模块自定义函数读取图片（BGR格式）
            img_bgr = predict.imreadex(self.pic_path)

            # 备份当前图片和原始图片数据
            self.current_image = img_bgr.copy()
            self.original_image = img_bgr.copy()

            # 转换为Tkinter格式图像，更新界面显示
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ctl.configure(image=self.imgtk)

            # 调用预测器进行车牌识别，返回识别结果、ROI和颜色
            result = self.predictor.predict(img_bgr)
            if result:
                r, roi, color = result
            else:
                r = None
                roi = None
                color = None

            # 显示识别结果和车牌区域
            self.show_roi(r, roi, color)

    def update_image(self, new_image):
        """更新显示的图像"""
        self.current_image = new_image.copy()
        self.imgtk = self.get_imgtk(new_image)
        self.image_ctl.configure(image=self.imgtk)

    def process_with_another_image(self, operation):
        """处理需要两张图片的操作"""
        if self.current_image is None:
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
        test_img_dir = os.path.join(base_dir, "test_image")  # 拼接成绝对路径
        file_path = askopenfilename(title="选择第二张图片", filetypes=[("jpg图片", "*.jpg")],
                                    initialdir=test_img_dir if os.path.exists(test_img_dir) else base_dir)
        if file_path:
            img2 = cv2.imread(file_path)
            if img2 is not None:
                # 调整第二张图片的大小以匹配当前图片
                img2 = cv2.resize(img2, (self.current_image.shape[1], self.current_image.shape[0]))
                if operation == "add":
                    result = self.image_processor.add_images(self.current_image, img2)
                elif operation == "subtract":
                    result = self.image_processor.subtract_images(self.current_image, img2)
                elif operation == "and":
                    result = self.image_processor.logical_and(self.current_image, img2)
                elif operation == "or":
                    result = self.image_processor.logical_or(self.current_image, img2)
                self.update_image(result)

    def reset_image(self):
        """恢复到原始加载的图像"""
        if self.original_image is not None:
            self.update_image(self.original_image)
            # 2) 将亮度滑块复位到 0
            self.brightness_var.set(0)
            # 3) 将对比度滑块复位到 1.0
            self.contrast_var.set(1.0)

    def add_image_processing_controls(self, frame_right1):
        """添加图像处理控件"""
        # 图像运算区域
        process_frame = ttk.LabelFrame(frame_right1, text='图像运算')
        process_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(process_frame, text='图像加法', command=lambda: self.process_with_another_image("add")).pack(
            fill='x', padx=5, pady=2)
        ttk.Button(process_frame, text='图像减法', command=lambda: self.process_with_another_image("subtract")).pack(
            fill='x', padx=5, pady=2)
        ttk.Button(process_frame, text='逻辑与', command=lambda: self.process_with_another_image("and")).pack(fill='x',
                                                                                                              padx=5,
                                                                                                              pady=2)
        ttk.Button(process_frame, text='逻辑或', command=lambda: self.process_with_another_image("or")).pack(fill='x',
                                                                                                             padx=5,
                                                                                                             pady=2)

        # 图像增强区域
        enhance_frame = ttk.LabelFrame(frame_right1, text='图像增强')
        enhance_frame.pack(fill='x', padx=5, pady=5)

        # 亮度调节
        # 亮度调节
        ttk.Label(enhance_frame, text='亮度调节:').pack(fill='x', padx=5)
        self.brightness_var = tk.IntVar(value=0)
        brightness_scale = ttk.Scale(enhance_frame, from_=-100, to=100, variable=self.brightness_var,
                                     orient='horizontal')
        brightness_scale.pack(fill='x', padx=5)
        ttk.Label(enhance_frame, textvariable=self.brightness_var).pack(pady=(0, 5))
        brightness_scale.bind('<ButtonRelease-1>', lambda e: self.update_image(
            self.image_processor.adjust_brightness(self.original_image, self.brightness_var.get())))

        # 对比度调节
        ttk.Label(enhance_frame, text='对比度调节:').pack(fill='x', padx=5)
        self.contrast_var = tk.DoubleVar(value=1.0)
        contrast_scale = ttk.Scale(enhance_frame, from_=0.1, to=3.0, variable=self.contrast_var, orient='horizontal')
        contrast_scale.pack(fill='x', padx=5)
        ttk.Label(enhance_frame, textvariable=self.contrast_var).pack(pady=(0, 5))
        contrast_scale.bind('<ButtonRelease-1>', lambda e: self.update_image(
            self.image_processor.adjust_contrast(self.original_image, self.contrast_var.get())))

        ttk.Button(enhance_frame, text='直方图均衡化',
                   command=lambda: self.update_image(self.image_processor.equalize_histogram(self.current_image))).pack(
            fill='x', padx=5, pady=2)

        # 图像变换区域
        transform_frame = ttk.LabelFrame(frame_right1, text='图像变换')
        transform_frame.pack(fill='x', padx=5, pady=5)

        # 旋转控制
        rotation_frame = ttk.Frame(transform_frame)
        rotation_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(rotation_frame, text='旋转角度:').pack(side='left')
        self.rotation_var = tk.StringVar(value='0')
        rotation_entry = ttk.Entry(rotation_frame, textvariable=self.rotation_var, width=10)
        rotation_entry.pack(side='left', padx=5)
        ttk.Button(rotation_frame, text='旋转',
                   command=lambda: self.update_image(
                       self.image_processor.rotate_image(self.current_image, float(self.rotation_var.get())))).pack(
            side='left')

        ttk.Button(transform_frame, text='水平镜像',
                   command=lambda: self.update_image(
                       self.image_processor.mirror_image(self.current_image, 'horizontal'))).pack(fill='x', padx=5,
                                                                                                  pady=2)
        ttk.Button(transform_frame, text='垂直镜像',
                   command=lambda: self.update_image(
                       self.image_processor.mirror_image(self.current_image, 'vertical'))).pack(fill='x', padx=5,
                                                                                                pady=2)

        # 图像滤波区域
        filter_frame = ttk.LabelFrame(frame_right1, text='图像滤波')
        filter_frame.pack(fill='x', padx=5, pady=5)

        # 核大小选择
        kernel_frame = ttk.Frame(filter_frame)
        kernel_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(kernel_frame, text='核大小:').pack(side='left')
        self.kernel_size_var = tk.StringVar(value='3')
        kernel_combo = ttk.Combobox(kernel_frame, textvariable=self.kernel_size_var, values=['3', '5', '7', '9'],
                                    width=5)
        kernel_combo.pack(side='left', padx=5)

        ttk.Button(filter_frame, text='均值滤波',
                   command=lambda: self.update_image(
                       self.image_processor.mean_filter(self.current_image, int(self.kernel_size_var.get())))).pack(
            fill='x', padx=5, pady=2)
        ttk.Button(filter_frame, text='中值滤波',
                   command=lambda: self.update_image(
                       self.image_processor.median_filter(self.current_image, int(self.kernel_size_var.get())))).pack(
            fill='x', padx=5, pady=2)
        ttk.Button(filter_frame, text='高斯滤波',
                   command=lambda: self.update_image(
                       self.image_processor.gaussian_filter(self.current_image, int(self.kernel_size_var.get())))).pack(
            fill='x', padx=5, pady=2)

        # 形态学处理区域
        morphology_frame = ttk.LabelFrame(frame_right1, text='形态学处理')
        morphology_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(morphology_frame, text='腐蚀',
                   command=lambda: self.update_image(
                       self.image_processor.erode(self.current_image, int(self.kernel_size_var.get())))).pack(fill='x',
                                                                                                              padx=5,
                                                                                                              pady=2)
        ttk.Button(morphology_frame, text='膨胀',
                   command=lambda: self.update_image(
                       self.image_processor.dilate(self.current_image, int(self.kernel_size_var.get())))).pack(fill='x',
                                                                                                               padx=5,
                                                                                                               pady=2)
        ttk.Button(morphology_frame, text='开运算',
                   command=lambda: self.update_image(
                       self.image_processor.opening(self.current_image, int(self.kernel_size_var.get())))).pack(
            fill='x', padx=5, pady=2)
        ttk.Button(morphology_frame, text='闭运算',
                   command=lambda: self.update_image(
                       self.image_processor.closing(self.current_image, int(self.kernel_size_var.get())))).pack(
            fill='x', padx=5, pady=2)

        # 边缘检测区域
        edge_frame = ttk.LabelFrame(frame_right1, text='边缘检测')
        edge_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(edge_frame, text='Canny边缘检测',
                   command=lambda: self.update_image(
                       self.image_processor.canny_edge(self.current_image, 100, 200))).pack(fill='x', padx=5, pady=2)
        ttk.Button(edge_frame, text='Hough直线检测',
                   command=lambda: self.update_image(self.image_processor.hough_lines(self.current_image))).pack(
            fill='x', padx=5, pady=2)
        ttk.Button(edge_frame, text='轮廓提取与分析',
                   command=lambda: self.update_image(
                       self.image_processor.find_and_draw_contours(self.current_image))).pack(fill='x', padx=5, pady=2)


def close_window():
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()
    surface = Surface(win)
    surface.pack(fill=tk.BOTH, expand=tk.YES)  # 确保 Surface 实例填充整个窗口
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()
