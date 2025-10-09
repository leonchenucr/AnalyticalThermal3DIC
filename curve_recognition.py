"""
curve_recognition.py - 轮廓识别模块
原始文件: Curve recognition.py
修改内容: 封装为函数 + 参数化 + 添加目录创建
核心算法: 100%保留
"""
import cv2
import numpy as np
import os


def extract_contours(image_path, output_path, domain_width=0.024, domain_height=0.024):
    """
    从芯片布局图像中提取轮廓点
    
    参数:
        image_path: 图像文件路径
        output_path: 输出轮廓点文件路径
        domain_width: 域宽度 (m) - 对应原代码的 w
        domain_height: 域高度 (m) - 对应原代码的 h
    
    返回:
        coords: 轮廓坐标数组
    
    修改说明:
        1. 将主逻辑封装为函数
        2. 硬编码的路径改为参数
        3. 添加 os.makedirs 确保输出目录存在
        4. 添加错误处理和日志输出
        5. 核心算法完全保留(第18-36行)
    """
    print(f"[CurveRecognition] 正在处理图像: {image_path}")
    
    # ========== 核心算法开始 (原代码第7-8行) ==========
    # Read image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 添加错误检查(原代码未包含)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    # 原代码第9-10行
    h = domain_height  # 原: h = 0.024
    w = domain_width   # 原: w = 0.024
    
    print(f"  图像尺寸: {image.shape}")
    
    # 原代码第12-14行 - 完全保留
    # Perform edge detection
    edges = cv2.Canny(image, 100, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 添加检查(原代码未包含)
    if not contours:
        raise ValueError("未找到有效轮廓")
    
    # 原代码第16-17行 - 完全保留
    # Assume only one main contour, select the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    print(f"  检测到轮廓点数: {len(contour)}")
    
    # 原代码第19-23行 - 完全保留
    # Get contour coordinates and convert to actual coordinates
    coords = contour.reshape(-1, 2)
    
    # Convert coordinates to float and scale
    coords = coords.astype(float)
    coords[:, 0] = coords[:, 0] / image.shape[1] * w
    coords[:, 1] = (image.shape[0] - coords[:, 1]) / image.shape[0] * h
    # ========== 核心算法结束 ==========
    
    # 添加目录创建(原代码未包含)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 原代码第25-28行 - 完全保留
    # Save coordinates to txt file
    with open(output_path, 'w') as f:
        for x, y in coords:
            f.write(f"{x:.6f}, {y:.6f}\n")
    
    # 原代码第30行 - 修改为参数化路径
    print(f"  轮廓点已保存: {output_path}")  # 原: print("Coordinates saved to points.txt")
    
    return coords


def main():
    """
    主函数 - 用于独立测试
    对应原代码的全局执行部分
    """
    # 原代码的硬编码路径
    extract_contours(
        image_path='E:/hot/multi-layer/shuju/s8_c1.jpg',
        output_path='E:/hot/multi-layer/shuju/s8_curve_points1.txt',
        domain_width=0.024,
        domain_height=0.024
    )
    print("轮廓识别完成!")


if __name__ == '__main__':
    main()


"""
========== 修改对比 ==========

原始代码结构:
--------------
import cv2
import numpy as np

image = cv2.imread('E:/hot/multi-layer/shuju/s8_c1.jpg', cv2.IMREAD_GRAYSCALE)
h = 0.024
w = 0.024
edges = cv2.Canny(image, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)
coords = contour.reshape(-1, 2)
coords = coords.astype(float)
coords[:, 0] = coords[:, 0] / image.shape[1] * w
coords[:, 1] = (image.shape[0] - coords[:, 1]) / image.shape[0] * h
with open('E:/hot/multi-layer/shuju/s8_curve_points1.txt', 'w') as f:
    for x, y in coords:
        f.write(f"{x:.6f}, {y:.6f}\n")
print("Coordinates saved to points.txt")

修改后结构:
-----------
def extract_contours(image_path, output_path, domain_width, domain_height):
    # 所有核心算法代码保持不变
    # 只是从全局作用域移到函数内
    # 硬编码路径改为参数
    pass

def main():
    # 原来的全局执行逻辑
    extract_contours(...)

if __name__ == '__main__':
    main()

核心算法变化: 0%
结构变化: 封装为函数
"""