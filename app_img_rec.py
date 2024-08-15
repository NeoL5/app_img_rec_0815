# -*- coding: utf-8 -*-
# @Author  : Zhiyi Leung
# @Time    : 2024/8/13 16:09
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np


ANGLE_THRESHOLD = 10  # 角度阈值，滤除接近水平的线条

st.title("Real-time object detection😍")
st.subheader("This app allow you to play with image filters")
# 侧边栏滑动条
blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
# 侧边栏复选框
apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')


# 高斯模糊
def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img


# 细节增强
def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr


def video_frame_callback(frame):
    # 将frame转换为NumPy数组img，转换后包含(height, width, 3)
    img = frame.to_ndarray(format="bgr24")

    processed_image = blur_image(img, blur_rate)

    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    # 进行边缘检测
    edges = cv2.Canny(gray_image, 50, 150)
    # 使用霍夫变换检测线条
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    # 检查是否检测到线条
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算线条的角度
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi

            # 转换为相对于水平面的角度
            if angle < 0:
                angle += 180
            if angle > 90:
                angle = 180 - angle

            # 过滤掉接近水平的线条
            if angle < ANGLE_THRESHOLD:
                continue

            # 绘制检测到的线条
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # 将角度标注在图像上
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2  # 计算线段的中点
            text = f'{angle:.1f}'
            cv2.putText(img, text, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return av.VideoFrame.from_ndarray(img, format="bgr24")  # 重设为BRG24格式


def main_loop():
    # 指定回调函数，视频流中的每一帧被捕获时调用。
    webrtc_streamer(key="object-detection",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True)


if __name__ == '__main__':
    main_loop()
