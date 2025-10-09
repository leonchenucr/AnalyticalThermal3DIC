"""
thermal_solver_3d.py - 3D热求解模块
原始文件: 3d_11layer.py
重命名为符合Python命名规范的文件名
内容与 3d_11layer.py 完全相同，只是文件名改变

使用此文件的优点:
1. 符合Python命名规范（文件名不以数字开头）
2. 导入更简单: from thermal_solver_3d import solve_thermal
3. 避免特殊导入方法

如果使用此文件，请:
1. 删除或重命名 3d_11layer.py
2. 使用此文件替代
3. 在GUI中直接导入: from thermal_solver_3d import solve_thermal
"""

# 以下内容与 3d_11layer.py 完全相同
import numpy as np
from scipy.sparse import csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time
from numba import jit
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Basic parameter settings
a = 0.024
b = 0.024
n_l = 11   # Number of layers
n_c = 6    # Layer containing heat source

# Absolute coordinates of layer top surfaces
th = np.array([0.0008, 0.0009, 0.0012, 0.0013, 0.00131, 0.00206, 0.00221, 0.00521, 0.00531, 0.00731, 0.01031])

# Anisotropic thermal conductivity parameters [kx, ky, kz]
ka_anisotropic = np.array([
    [2, 2, 0.4],   
    [29.75, 29.75, 35.36],   
    [102, 102, 61.50],   
    [10, 10, 80.275],   
    [1.5, 1.5, 1.5],   
    [140, 140, 140],  
    [30, 30, 30],     
    [400, 400, 400],   
    [10, 10, 10],     
    [400, 400, 400],   
    [400, 400, 400]  
])

# Extract thermal conductivity in each direction
kx = ka_anisotropic[:, 0]
ky = ka_anisotropic[:, 1]
kz = ka_anisotropic[:, 2]

# Pre-compute π values
pi = np.pi
pi_a = pi / a
pi_b = pi / b


@jit(nopython=True)
def compute_integral_segment(start, end, L, mode):
    if mode == 0:
        return end - start
    k = mode * pi / L
    return (L / (mode * pi)) * (np.sin(k * end) - np.sin(k * start))


def calculate_gmn(x_starts, x_ends, y_starts, y_ends, power_densities, num_eigen):
    gmn = np.zeros((num_eigen, num_eigen))
    
    int_x_all = np.array([[compute_integral_segment(x_s, x_e, a, p) 
                          for p in range(num_eigen)] 
                         for x_s, x_e in zip(x_starts, x_ends)])
    
    int_y_all = np.array([[compute_integral_segment(y_s, y_e, b, k) 
                          for k in range(num_eigen)] 
                         for y_s, y_e in zip(y_starts, y_ends)])
    
    norm_x = np.where(np.arange(num_eigen) == 0, 1/a, 2/a)
    norm_y = np.where(np.arange(num_eigen) == 0, 1/b, 2/b)
    
    for idx in range(len(power_densities)):
        q = power_densities[idx]
        outer = np.outer(int_x_all[idx] * norm_x, int_y_all[idx] * norm_y)
        gmn += q * outer
    
    return gmn


def calculate_fmn(data_f, num_eigen):
    num_l = 400
    x = (np.linspace(0, a, num_l+1)[:-1] + np.linspace(0, a, num_l+1)[1:])/2
    y = (np.linspace(0, b, num_l+1)[:-1] + np.linspace(0, b, num_l+1)[1:])/2
    
    cos_mx = np.array([np.cos(m * pi_a * x) for m in range(num_eigen)])
    cos_ny = np.array([np.cos(n * pi_b * y) for n in range(num_eigen)])
    
    fx = np.zeros((num_eigen, num_l))
    for m in range(num_eigen):
        norm_factor = (2/a if m > 0 else 1/a)
        fx[m] = np.sum(data_f * cos_mx[m].reshape(1, -1), axis=1) * (a/num_l) * norm_factor
    
    fmn = np.zeros((num_eigen, num_eigen))
    for m in range(num_eigen):
        for n in range(num_eigen):
            norm_factor = (2/b if n > 0 else 1/b)
            fmn[m, n] = np.sum(fx[m] * cos_ny[n]) * (b/num_l) * norm_factor
    
    return fmn


def calculate_lambda_mn(m, n, layer_idx):
    kx_layer = kx[layer_idx]
    ky_layer = ky[layer_idx] 
    kz_layer = kz[layer_idx]
    
    lambda_mn_squared = (kx_layer/kz_layer) * (m * pi_a)**2 + (ky_layer/kz_layer) * (n * pi_b)**2
    return np.sqrt(lambda_mn_squared)


def solve_system(m, n, gmn, fmn):
    if m == 0 and n == 0:
        amatrix = lil_matrix((n_l*2, n_l*2))
        bvector = lil_matrix((n_l*2, 1))
        
        amatrix[0, 1] = 1
        bvector[0, 0] = 0
        
        for i in range(1, n_l):
            amatrix[2*i-1, 2*i-1] = kz[i-1]
            amatrix[2*i-1, 2*i+1] = -kz[i]
            
            amatrix[2*i, 2*i-2] = 1
            amatrix[2*i, 2*i-1] = th[i-1]
            amatrix[2*i, 2*i] = -1
            amatrix[2*i, 2*i+1] = -th[i-1]
        
        amatrix[2*n_l-1, 2*n_l-2] = 1
        amatrix[2*n_l-1, 2*n_l-1] = th[n_l-1]
        bvector[2*n_l-1, 0] = fmn[0,0]
        
        bvector[2*(n_c-1)-1, 0] = -gmn[0,0]*th[n_c-2]
        bvector[2*(n_c-1), 0] = -0.5*gmn[0,0]/kz[n_c-1]*th[n_c-2]**2
        bvector[2*(n_c)-1, 0] = gmn[0,0]*th[n_c-1]
        bvector[2*(n_c), 0] = 0.5*gmn[0,0]/kz[n_c-1]*th[n_c-1]**2
        
    else:
        amatrix = lil_matrix((n_l*2, n_l*2))
        bvector = lil_matrix((n_l*2, 1))
        
        amatrix[0, 0] = 1
        amatrix[0, 1] = -1
        bvector[0, 0] = 0
        
        for i in range(1, n_l):
            lamb_prev = calculate_lambda_mn(m, n, i-1)
            lamb_curr = calculate_lambda_mn(m, n, i)
            
            amatrix[2*i-1, 2*i-2] = -kz[i-1] * lamb_prev * np.exp(-lamb_prev*th[i-1])
            amatrix[2*i-1, 2*i-1] = kz[i-1] * lamb_prev * np.exp(lamb_prev*th[i-1])
            amatrix[2*i-1, 2*i] = kz[i] * lamb_curr * np.exp(-lamb_curr*th[i-1])
            amatrix[2*i-1, 2*i+1] = -kz[i] * lamb_curr * np.exp(lamb_curr*th[i-1])
            
            amatrix[2*i, 2*i-2] = np.exp(-lamb_prev*th[i-1])
            amatrix[2*i, 2*i-1] = np.exp(lamb_prev*th[i-1])
            amatrix[2*i, 2*i] = -np.exp(-lamb_curr*th[i-1])
            amatrix[2*i, 2*i+1] = -np.exp(lamb_curr*th[i-1])
        
        lamb_top = calculate_lambda_mn(m, n, n_l-1)
        amatrix[2*n_l-1, 2*n_l-2] = np.exp(-lamb_top*th[n_l-1])
        amatrix[2*n_l-1, 2*n_l-1] = np.exp(lamb_top*th[n_l-1])
        bvector[2*n_l-1, 0] = fmn[m,n]
        
        lamb_source = calculate_lambda_mn(m, n, n_c-1)
        bvector[2*(n_c-1), 0] = gmn[m,n]/(kz[n_c-1]*lamb_source**2)
        bvector[2*(n_c), 0] = -gmn[m,n]/(kz[n_c-1]*lamb_source**2)
    
    return spsolve(amatrix.tocsc(), bvector.tocsc())


def calculate_temperature(AB, gmn, x, y, z, num_eigen):
    layer_ends = th
    layer_idx = np.searchsorted(layer_ends, z, side='left')
    layer_idx = np.clip(layer_idx, 0, len(layer_ends)-1)
    
    temp = np.zeros_like(z, dtype=float)
    
    A_0 = AB[0, 0, 2 * layer_idx]
    B_0 = AB[0, 0, 2 * layer_idx + 1]
    temp += A_0 + B_0 * z
    
    mask = layer_idx == (n_c-1)
    if np.any(mask):
        kz_source = kz[n_c-1]
        temp[mask] -= 0.5 * gmn[0, 0] / kz_source * z[mask]**2
    
    for m in range(num_eigen):
        for n in range(num_eigen):
            if m == 0 and n == 0:
                continue
            
            unique_layers = np.unique(layer_idx)
            for layer in unique_layers:
                layer_mask = layer_idx == layer
                if not np.any(layer_mask):
                    continue
                
                lamb = calculate_lambda_mn(m, n, layer)
                A = AB[m, n, 2 * layer]
                B = AB[m, n, 2 * layer + 1]
                
                z_layer = z[layer_mask]
                x_layer = x[layer_mask]
                y_layer = y[layer_mask]
                
                exp_neg = np.exp(-lamb * z_layer)
                exp_pos = np.exp(lamb * z_layer)
                cos_mx = np.cos(m * pi_a * x_layer)
                cos_ny = np.cos(n * pi_b * y_layer)
                
                temp[layer_mask] += (A * exp_neg + B * exp_pos) * cos_mx * cos_ny
                
                if layer == (n_c-1):
                    temp[layer_mask] += gmn[m, n] / (kz[layer] * lamb**2) * cos_mx * cos_ny

    return temp


def generate_visualizations(x, y, z, temp_calc, temp_comsol, error, output_dir):
    dx = x.max() - x.min()
    dy = y.max() - y.min()
    dz = z.max() - z.min()
    aspect_ratio = [dx, dy, dz]
    
    # 1. 计算温度场
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.view_init(elev=30, azim=225)
    
    sc1 = ax1.scatter(x, y, z, c=temp_calc, cmap='jet', s=10, alpha=0.6)
    cbar1 = fig1.colorbar(sc1, ax=ax1, shrink=0.5, aspect=20, pad=0.08)
    cbar1.set_label('Temperature (K)', fontsize=12, rotation=90, labelpad=15)
    
    ax1.set_title('Proposed Method', fontsize=14)
    ax1.set_xlabel('x (m)', fontsize=12)
    ax1.set_ylabel('y (m)', fontsize=12)
    ax1.set_zlabel('z (m)', fontsize=12, rotation=90)
    ax1.set_box_aspect(aspect_ratio)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calculated_temperature_mor_3d.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. COMSOL温度场
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.view_init(elev=30, azim=225)
    
    sc2 = ax2.scatter(x, y, z, c=temp_comsol, cmap='jet', s=10, alpha=0.6)
    cbar2 = fig2.colorbar(sc2, ax=ax2, shrink=0.5, aspect=20, pad=0.08)
    cbar2.set_label('Temperature (K)', fontsize=12, rotation=90, labelpad=15)
    
    ax2.set_title('COMSOL', fontsize=14)
    ax2.set_xlabel('x (m)', fontsize=12)
    ax2.set_ylabel('y (m)', fontsize=12)
    ax2.set_zlabel('z (m)', fontsize=12, rotation=90)
    ax2.set_box_aspect(aspect_ratio)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comsol_temperature_3d.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 误差分布
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.view_init(elev=30, azim=225)
    
    sc3 = ax3.scatter(x, y, z, c=error, cmap='jet', s=10, alpha=0.6, vmin=0)
    cbar3 = fig3.colorbar(sc3, ax=ax3, shrink=0.5, aspect=20, pad=0.08)
    cbar3.set_label('Absolute Error (K)', fontsize=12, rotation=90, labelpad=15)
    
    ax3.set_title('Error Distribution', fontsize=14)
    ax3.set_xlabel('x (m)', fontsize=12)
    ax3.set_ylabel('y (m)', fontsize=12)
    ax3.set_zlabel('z (m)', fontsize=12, rotation=90)
    ax3.set_box_aspect(aspect_ratio)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chip_layer_error_distribution.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  可视化结果已保存")


def generate_dummy_plots(output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Temperature Field\n(No COMSOL data for comparison)', 
           ha='center', va='center', fontsize=16)
    ax.axis('off')
    plt.savefig(os.path.join(output_dir, 'calculated_temperature_mor_3d.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'COMSOL Data Not Provided', 
           ha='center', va='center', fontsize=16)
    ax.axis('off')
    plt.savefig(os.path.join(output_dir, 'comsol_temperature_3d.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'Error Analysis\n(No COMSOL data for comparison)', 
           ha='center', va='center', fontsize=16)
    ax.axis('off')
    plt.savefig(os.path.join(output_dir, 'chip_layer_error_distribution.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def solve_thermal(mesh_file, boundary_file=None, comsol_file=None, 
                 output_dir='./outputs', num_eigen=30):
    """
    执行3D热求解
    
    参数:
        mesh_file: 网格文件路径
        boundary_file: 边界条件文件路径(可选)
        comsol_file: COMSOL验证数据路径(可选)
        output_dir: 输出目录
        num_eigen: 特征模式数
    
    返回:
        results: 包含误差统计和温度结果的字典
    """
    print(f"[ThermalSolver] 开始3D热场计算 (特征模式数={num_eigen})")
    start_time = time.time()
    
    print("  加载网格数据...")
    mesh_data = np.loadtxt(mesh_file, delimiter=',', skiprows=1)
    
    x_starts = mesh_data[:, 2]
    x_ends = mesh_data[:, 3]
    y_starts = mesh_data[:, 4]
    y_ends = mesh_data[:, 5]
    power_densities = mesh_data[:, 6]
    
    print(f"  网格单元数: {len(mesh_data)}")
    
    print("  计算热源傅里叶系数...")
    gmn = calculate_gmn(x_starts, x_ends, y_starts, y_ends, power_densities, num_eigen)
    
    print("  加载边界条件...")
    if boundary_file and os.path.exists(boundary_file):
        data_f = np.loadtxt(boundary_file)
        fmn = calculate_fmn(data_f, num_eigen)
    else:
        print("  警告: 未提供边界条件,使用零边界")
        fmn = np.zeros((num_eigen, num_eigen))
    
    print("  求解线性系统...")
    AB = np.zeros((num_eigen, num_eigen, n_l*2))
    for m in range(num_eigen):
        for n in range(num_eigen):
            AB[m,n,:] = solve_system(m, n, gmn, fmn)
        if (m + 1) % 10 == 0:
            print(f"    已完成 {m+1}/{num_eigen} 模态")
    
    calc_time = time.time() - start_time
    print(f"  计算完成,用时: {calc_time:.2f} 秒")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {'calc_time': calc_time}
    
    if comsol_file and os.path.exists(comsol_file):
        print("  计算与COMSOL的误差...")
        comsol_data = np.loadtxt(comsol_file)
        
        x_comsol = comsol_data[:, 0]
        y_comsol = comsol_data[:, 1]
        z_comsol = comsol_data[:, 2]
        temp_comsol = comsol_data[:, 3]
        
        temp_calculated = calculate_temperature(AB, gmn, x_comsol, y_comsol, z_comsol, num_eigen)
        
        error = np.abs(temp_calculated - temp_comsol)
        results['mean_error'] = float(np.mean(error))
        results['max_error'] = float(np.max(error))
        results['max_temp'] = float(np.max(temp_calculated))
        results['min_temp'] = float(np.min(temp_calculated))
        
        print(f"  平均误差: {results['mean_error']:.4f} K")
        print(f"  最大误差: {results['max_error']:.4f} K")
        
        print("  生成可视化结果...")
        generate_visualizations(x_comsol, y_comsol, z_comsol,
                               temp_calculated, temp_comsol, error, output_dir)
    else:
        print("  未提供COMSOL数据,跳过验证")
        results['max_temp'] = 0.0
        results['min_temp'] = 298.0
        results['mean_error'] = 0.0
        results['max_error'] = 0.0
        
        generate_dummy_plots(output_dir)
    
    return results


def main():
    """主函数 - 用于独立测试"""
    solve_thermal(
        mesh_file='E:/hot/multi-layer/shuju/s8_pd_test_S.txt',
        boundary_file='E:/hot/multi-layer/shuju/combined_fd3.txt',
        comsol_file='E:/hot/multi-layer/V2/sr2.7gengxi.txt',
        output_dir='E:/hot/multi-layer/shuju',
        num_eigen=30
    )
    print("热求解完成!")


if __name__ == "__main__":
    main()