#!/usr/bin/python
# -*- coding: utf-8 -*-
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2019 Intel Corporation. All Rights Reserved.
# Python 2/3 compatibility
from __future__ import print_function

"""
This example shows how to use T265 intrinsics and extrinsics in OpenCV to
asynchronously compute depth maps from T265 fisheye images on the host.

T265 is not a depth camera and the quality of passive-only depth options will
always be limited compared to (e.g.) the D4XX series cameras. However, T265 does
have two global shutter cameras in a stereo configuration, and in this example
we show how to set up OpenCV to undistort the images and compute stereo depth
from them.

Getting started with python3, OpenCV and T265 on Ubuntu 16.04:

First, set up the virtual enviroment:

$ apt-get install python3-venv  # install python3 built in venv support
$ python3 -m venv py3librs      # create a virtual environment in pylibrs
$ source py3librs/bin/activate  # activate the venv, do this from every terminal
$ pip install opencv-python     # install opencv 4.1 in the venv
$ pip install pyrealsense2      # install librealsense python bindings

Then, for every new terminal:

$ source py3librs/bin/activate  # Activate the virtual environment
$ python3 t265_stereo.py        # Run the example
此示例显示如何在OpenCV中使用T265内部和外部

从主机上的T265鱼眼图像异步计算深度图。



T265不是深度相机，仅被动深度选项的质量将

与（例如）D4XX系列相机相比总是受到限制。然而，T265确实如此

在立体配置中有两个全局快门相机，在本例中

我们展示了如何设置OpenCV来消除图像失真并计算立体深度

从他们那里。



在Ubuntu 16.04上开始使用python3、OpenCV和T265：



首先，设置虚拟环境：



$apt-get-install python3 venv#安装内置于venv支持的python

$python3-m venv py3librs#在pylibrs中创建虚拟环境

$source py3librs/bin/activate#激活venv，从每个终端执行此操作

$pip install opencv python#在venv中安装opencv 4.1

$pip install pyrealsense2#安装librealsense python绑定



然后，对于每个新终端：



$source py3librs/bin/activate#激活虚拟环境

$python3 t265_stereo。py#运行示例
"""

# First import the library
import pyrealsense2 as rs

# Import OpenCV and numpy
import cv2
import numpy as np
from math import tan, pi

"""
In this section, we will set up the functions that will translate the camera
intrinsics and extrinsics from librealsense into parameters that can be used
with OpenCV.

The T265 uses very wide angle lenses, so the distortion is modeled using a four
parameter distortion model known as Kanalla-Brandt. OpenCV supports this
distortion model in their "fisheye" module, more details can be found here:

https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
在本节中，我们将设置平移相机的功能
从librrealsense到可以使用的参数的内在和外在
使用OpenCV。

T265使用非常宽的角度镜头，因此使用四个
被称为Kanalla-Brandt的参数失真模型。OpenCV支持此功能
在他们的“鱼眼”模块中的失真模型，可以在这里找到更多细节：

https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
"""

"""
Returns R, T transform from src to dst
返回从src到dst的R，T转换
"""
def get_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3,3]).T
    T = np.array(extrinsics.translation)
    return (R, T)

"""
Returns a camera matrix K from librealsense intrinsics
返回从src到dst的R，T转换
"""
def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                     [            0, intrinsics.fy, intrinsics.ppy],
                     [            0,             0,              1]])

"""
Returns the fisheye distortion from librealsense intrinsics
从librrealsense内部函数返回鱼眼失真
"""
def fisheye_distortion(intrinsics):
    return np.array(intrinsics.coeffs[:4])

# Set up a mutex to share data between threads 设置互斥体以在线程之间共享数据
from threading import Lock
frame_mutex = Lock()
frame_data = {"left"  : None,
              "right" : None,
              "timestamp_ms" : None
              }

"""
This callback is called on a separate thread, so we must use a mutex
to ensure that data is synchronized properly. We should also be
careful not to do much work on this thread to avoid data backing up in the
callback queue.
This callback is called on a separate thread, so we must use a mutex
to ensure that data is synchronized properly. We should also be
careful not to do much work on this thread to avoid data backing up in the
callback queue.
This callback is called on a separate thread, so we must use a mutex
to ensure that data is synchronized properly. We should also be
careful not to do much work on this thread to avoid data backing up in the
callback queue.
这个回调是在单独的线程上调用的，所以我们必须使用互斥锁来确保数据正确同步。我们还应该注意不要在这个线程上做太多工作，以避免回调队列中的数据备份。
"""
def callback(frame):
    global frame_data
    if frame.is_frameset():
        frameset = frame.as_frameset()
        f1 = frameset.get_fisheye_frame(1).as_video_frame()
        f2 = frameset.get_fisheye_frame(2).as_video_frame()
        left_data = np.asanyarray(f1.get_data())
        right_data = np.asanyarray(f2.get_data())
        ts = frameset.get_timestamp()
        frame_mutex.acquire()
        frame_data["left"] = left_data
        frame_data["right"] = right_data
        frame_data["timestamp_ms"] = ts
        frame_mutex.release()

# Declare RealSense pipeline, encapsulating the actual device and sensors 声明RealSense管道，封装实际设备和传感器
pipe = rs.pipeline()

# Build config object and stream everything 构建配置对象并流式传输所有内容
cfg = rs.config()

# Start streaming with our callback 使用我们的回调开始流式传输
pipe.start(cfg, callback)

try:
    # Set up an OpenCV window to visualize the results 设置OpenCV窗口以可视化结果

    WINDOW_TITLE = 'Realsense'
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    # Configure the OpenCV stereo algorithm. See 配置OpenCV立体算法
    # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html for a
    # description of the parameters
    window_size = 5
    min_disp = 0
    # must be divisible by 16 必须可被16整除
    num_disp = 112 - min_disp
    max_disp = min_disp + num_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                   numDisparities = num_disp,
                                   blockSize = 16,
                                   P1 = 8*3*window_size**2,
                                   P2 = 32*3*window_size**2,
                                   disp12MaxDiff = 1,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 32)

    # Retreive the stream and intrinsic properties for both cameras 检索两个相机的流和固有属性
    profiles = pipe.get_active_profile()
    streams = {"left"  : profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
               "right" : profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
    intrinsics = {"left"  : streams["left"].get_intrinsics(),
                  "right" : streams["right"].get_intrinsics()}

    # Print information about both cameras 打印有关两个摄像头的信息
    print("Left camera:",  intrinsics["left"])
    print("Right camera:", intrinsics["right"])

    # Translate the intrinsics from librealsense into OpenCV 将librrealsense中的intrinsic转换为OpenCV
    K_left  = camera_matrix(intrinsics["left"])
    D_left  = fisheye_distortion(intrinsics["left"])
    K_right = camera_matrix(intrinsics["right"])
    D_right = fisheye_distortion(intrinsics["right"])
    (width, height) = (intrinsics["left"].width, intrinsics["left"].height)

    # Get the relative extrinsics between the left and right camera 获取左右摄像头之间的相对外部信息
    (R, T) = get_extrinsics(streams["left"], streams["right"])

    # We need to determine what focal length our undistorted images should have
    # in order to set up the camera matrices for initUndistortRectifyMap.  We
    # could use stereoRectify, but here we show how to derive these projection
    # matrices from the calibration and a desired height and field of view
    # 我们需要确定无失真图像的焦距，以便为initUndestorRectifyMap设置相机矩阵。我们可以使用立体矫正，但这里我们展示了如何从校准和期望的高度和视野中导出这些投影矩阵
    # We calculate the undistorted focal length:我们计算无失真焦距：
    #
    #         h
    # -----------------
    #  \      |      /
    #    \    | f  /
    #     \   |   /
    #      \ fov /
    #        \|/
    stereo_fov_rad = 90 * (pi/180)  # 90 degree desired fov 90度所需中心凹
    stereo_height_px = 300          # 300x300 pixel stereo output 300x300像素立体声输出
    stereo_focal_px = stereo_height_px/2 / tan(stereo_fov_rad/2)

    # We set the left rotation to identity and the right rotation
    # the rotation between the cameras 我们将左旋转设置为同一，右旋转设置为摄影机之间的旋转
    R_left = np.eye(3)
    R_right = R

    # The stereo algorithm needs max_disp extra pixels in order to produce valid
    # disparity on the desired output region. This changes the width, but the
    # center of projection should be on the center of the cropped image 
    # 立体算法需要max_disp额外像素，以便在期望的输出区域上产生有效的视差。这将更改宽度，但投影中心应位于裁剪图像的中心
    stereo_width_px = stereo_height_px + max_disp
    stereo_size = (stereo_width_px, stereo_height_px)
    stereo_cx = (stereo_height_px - 1)/2 + max_disp
    stereo_cy = (stereo_height_px - 1)/2

    # Construct the left and right projection matrices, the only difference is
    # that the right projection matrix should have a shift along the x axis of
    # baseline*focal_length 构造左投影矩阵和右投影矩阵，唯一的区别是右投影矩阵应该沿着基线*焦点长度的x轴移动
    P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                       [0, stereo_focal_px, stereo_cy, 0],
                       [0,               0,         1, 0]])
    P_right = P_left.copy()
    P_right[0][3] = T[0]*stereo_focal_px

    # Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
    # since we will crop the disparity later 构造用于cv2.reprojectImageTo3D的Q。从x中减去max_disp，因为我们稍后将裁剪视差
    Q = np.array([[1, 0,       0, -(stereo_cx - max_disp)],
                  [0, 1,       0, -stereo_cy],
                  [0, 0,       0, stereo_focal_px],
                  [0, 0, -1/T[0], 0]])

    # Create an undistortion map for the left and right camera which applies the
    # rectification and undoes the camera distortion. This only has to be done
    # once 为左右相机创建一个不失真贴图，该贴图应用校正并消除相机失真。这只需要做一次
    m1type = cv2.CV_32FC1
    (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, P_left, stereo_size, m1type)
    (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, P_right, stereo_size, m1type)
    undistort_rectify = {"left"  : (lm1, lm2),
                         "right" : (rm1, rm2)}

    mode = "stack"
    while True:
        # Check if the camera has acquired any frames 检查摄像头是否已获取任何帧
        frame_mutex.acquire()
        valid = frame_data["timestamp_ms"] is not None
        frame_mutex.release()

        # If frames are ready to process 如果帧已准备好处理
        if valid:
            # Hold the mutex only long enough to copy the stereo frames 保持互斥锁的时间仅足以复制立体声帧
            frame_mutex.acquire()
            frame_copy = {"left"  : frame_data["left"].copy(),
                          "right" : frame_data["right"].copy()}
            frame_mutex.release()

            # Undistort and crop the center of the frames 不失真并裁剪框架的中心
            center_undistorted = {"left" : cv2.remap(src = frame_copy["left"],
                                          map1 = undistort_rectify["left"][0],
                                          map2 = undistort_rectify["left"][1],
                                          interpolation = cv2.INTER_LINEAR),
                                  "right" : cv2.remap(src = frame_copy["right"],
                                          map1 = undistort_rectify["right"][0],
                                          map2 = undistort_rectify["right"][1],
                                          interpolation = cv2.INTER_LINEAR)}

            # compute the disparity on the center of the frames and convert it to a pixel disparity (divide by DISP_SCALE=16)
            # 计算帧中心的视差并将其转换为像素视差（除以DISP_SCALE=16）
            disparity = stereo.compute(center_undistorted["left"], center_undistorted["right"]).astype(np.float32) / 16.0

            # re-crop just the valid part of the disparity 重新裁剪差异的有效部分
            disparity = disparity[:,max_disp:]

            # convert disparity to 0-255 and color it 将视差转换为0-255并为其上色
            disp_vis = 255*(disparity - min_disp)/ num_disp
            disp_color = cv2.applyColorMap(cv2.convertScaleAbs(disp_vis,1), cv2.COLORMAP_JET)
            color_image_left = cv2.cvtColor(center_undistorted["left"][:,max_disp:], cv2.COLOR_GRAY2RGB)
            color_image_right = cv2.cvtColor(center_undistorted["right"][:,max_disp:], cv2.COLOR_GRAY2RGB)
            if mode == "all":
                cv2.imshow(WINDOW_TITLE, np.hstack((color_image_left, color_image_right,disp_color)))
            if mode == "stack":
                cv2.imshow(WINDOW_TITLE, np.hstack((color_image_left, disp_color)))
            if mode == "overlay":
                ind = disparity >= min_disp
                color_image_left[ind, 0] = disp_color[ind, 0]
                color_image_left[ind, 1] = disp_color[ind, 1]
                color_image_left[ind, 2] = disp_color[ind, 2]
                cv2.imshow(WINDOW_TITLE, color_image_left)
        key = cv2.waitKey(1)
        if key == ord('s'): mode = "stack"
        if key == ord('o'): mode = "overlay"
        if key == ord('a'): mode = "all"
        if key == ord('q') or cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
            break
finally:
    pipe.stop()
