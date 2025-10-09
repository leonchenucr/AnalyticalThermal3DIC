"""
rectangular_corresponding_power.py - 功率计算模块
原始文件: Rectangular corresponding power.py
修改内容: 封装为函数 + 参数化功率系数
核心算法: 100%保留
"""
import numpy as np
from numpy import genfromtxt
import os


def calculate_power(funit_file, output_power_file, power_coefficient=9.811258839738471508e+09):
    """
    计算每个矩形的功率
    
    参数:
        funit_file: 矩形参数文件路径
        output_power_file: 输出功率文件路径
        power_coefficient: 功率密度常数 (W/m³) - 原代码第16行的硬编码值
    
    返回:
        recalculated_power: 功率数组
    
    修改说明:
        1. 将主逻辑封装为函数
        2. 功率系数从硬编码改为参数
        3. 添加目录创建和统计输出
        4. 核心算法100%保留(第9-32行)
    """
    print(f"[PowerCalculator] 计算功率分布 (系数={power_coefficient:.2e})")
    
    # ========== 核心算法开始 (原代码第5-7行) ==========
    # Read data
    FUnit = genfromtxt(funit_file, dtype=str)  # 原: 'E:/hot/multi-layer/shuju/s8_FUnit1.txt'
    num_power = FUnit.shape[0]  # Number of rectangles
    
    # Initialize power list and name list
    recalculated_power = []
    rect_names = []
    
    # 原代码第9-32行 - 完全保留
    # Iterate through each rectangle
    for i in range(num_power):
        # Get rectangle name (assuming name is in column 0)
        rect_name = FUnit[i, 0]  
        rect_names.append(rect_name)
        
        # 原代码第15-19行 - 唯一修改是将硬编码值改为参数
        # Determine rectangle type and set constant value
        if rect_name.startswith("1ORIG"):  
            constant_value = power_coefficient  # 原: constant_value = 9.811258839738471508e+09
        elif rect_name.startswith("1EXT"):  
            constant_value = 0.0
        else:  
            constant_value = 0.0
        
        # 原代码第21-26行 - 完全保留
        # Get rectangle boundaries and dimensions (assuming corresponding columns are FUnit[i, 3], FUnit[i, 4], FUnit[i, 1], FUnit[i, 2])
        x_start = float(FUnit[i, 3])
        y_start = float(FUnit[i, 4])
        width = float(FUnit[i, 1])
        height = float(FUnit[i, 2])
        
        # Calculate rectangle area
        area = width * height
        
        # Calculate total power
        total_power = constant_value * area * 0.00015
        
        # Add to power list
        recalculated_power.append(total_power)
    # ========== 核心算法结束 ==========
    
    # 添加统计输出(原代码未包含)
    orig_count = sum(1 for name in rect_names if name.startswith("1ORIG"))
    total_power_sum = sum(p for p, name in zip(recalculated_power, rect_names) 
                         if name.startswith("1ORIG"))
    
    print(f"  热源矩形数: {orig_count}")
    print(f"  总功率: {total_power_sum:.4f} W")
    
    # 添加目录创建(原代码未包含)
    os.makedirs(os.path.dirname(output_power_file), exist_ok=True)
    
    # 原代码第34-38行 - 完全保留
    # Save rectangle names and power sources to file
    output_file = output_power_file  # 原: 'E:/hot/multi-layer/shuju/s8_power1.txt'
    with open(output_file, 'w') as f:
        # Write rectangle names
        f.write(' '.join(rect_names) + '\n')
        # Write power sources
        f.write(' '.join(map(str, recalculated_power)) + '\n')
    
    # 原代码第40行 - 修改为参数化路径
    print(f"  功率数据已保存: {output_power_file}")  
    # 原: print(f"Power calculation completed, results saved to file {output_file}")
    
    return recalculated_power


def main():
    """
    主函数 - 用于独立测试
    对应原代码的全局执行部分
    """
    calculate_power(
        funit_file='E:/hot/multi-layer/shuju/s8_FUnit1.txt',
        output_power_file='E:/hot/multi-layer/shuju/s8_power1.txt',
        power_coefficient=9.811258839738471508e+09  # 原代码第16行的值
    )
    print("功率计算完成!")


if __name__ == '__main__':
    main()


"""
========== 修改对比 ==========

原始代码结构(简化):
-------------------
import numpy as np
from numpy import genfromtxt

FUnit = genfromtxt('E:/hot/multi-layer/shuju/s8_FUnit1.txt', dtype=str)
num_power = FUnit.shape[0]
recalculated_power = []
rect_names = []

for i in range(num_power):
    rect_name = FUnit[i, 0]
    rect_names.append(rect_name)
    
    if rect_name.startswith("1ORIG"):
        constant_value = 9.811258839738471508e+09  # 硬编码
    elif rect_name.startswith("1EXT"):
        constant_value = 0.0
    else:
        constant_value = 0.0
    
    x_start = float(FUnit[i, 3])
    y_start = float(FUnit[i, 4])
    width = float(FUnit[i, 1])
    height = float(FUnit[i, 2])
    area = width * height
    total_power = constant_value * area * 0.00015
    recalculated_power.append(total_power)

output_file = 'E:/hot/multi-layer/shuju/s8_power1.txt'  # 硬编码
with open(output_file, 'w') as f:
    f.write(' '.join(rect_names) + '\n')
    f.write(' '.join(map(str, recalculated_power)) + '\n')
print(f"Power calculation completed, results saved to file {output_file}")

修改后结构:
-----------
def calculate_power(funit_file, output_power_file, power_coefficient):
    # 所有核心逻辑保持100%不变
    # 将硬编码的功率系数改为参数
    # 将硬编码的文件路径改为参数
    pass

def main():
    calculate_power(
        funit_file='E:/...',
        output_power_file='E:/...',
        power_coefficient=9.811258839738471508e+09
    )

主要变化:
1. ✅ 封装为函数
2. ✅ 参数化: 功率系数(第16行)、文件路径
3. ✅ 添加: 统计输出、目录创建
4. ✅ 核心算法: 100%保留

核心算法变化: 0%
唯一实质性修改: 第16行硬编码值 → 函数参数
"""