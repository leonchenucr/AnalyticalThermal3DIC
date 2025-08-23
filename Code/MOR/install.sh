#!/bin/bash

# MOR (Model Order Reduction) 热传导计算器安装脚本
# 适用于Linux/macOS系统

echo "🚀 开始安装MOR热传导计算器..."
echo "=================================="

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 1 ]]; then
    echo "✅ Python版本检查通过: $python_version"
else
    echo "❌ Python版本过低，需要3.8+，当前版本: $python_version"
    exit 1
fi

# 检查conda是否安装
if command -v conda &> /dev/null; then
    echo "✅ 检测到conda环境管理器"
    
    # 创建新的conda环境
    echo "🔧 创建conda环境 'mor_env'..."
    conda create -n mor_env python=3.9 -y
    
    # 激活环境
    echo "🔧 激活conda环境..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate mor_env
    
    # 安装依赖
    echo "📦 安装Python依赖包..."
    conda install -c conda-forge numpy scipy matplotlib mpi4py -y
    
    # 安装DOLFINx (可能需要特殊处理)
    echo "🔧 安装DOLFINx..."
    conda install -c conda-forge fenics-dolfinx -y
    
    # 安装GMSH
    echo "🔧 安装GMSH..."
    conda install -c conda-forge gmsh -y
    
else
    echo "⚠️  未检测到conda，使用pip安装..."
    
    # 创建虚拟环境
    echo "🔧 创建Python虚拟环境..."
    python3 -m venv mor_env
    
    # 激活环境
    echo "🔧 激活虚拟环境..."
    source mor_env/bin/activate
    
    # 升级pip
    pip install --upgrade pip
    
    # 安装依赖
    echo "📦 安装Python依赖包..."
    pip install -r requirements.txt
fi

echo ""
echo "🎉 安装完成！"
echo "=================================="
echo "使用方法："
echo "1. 激活环境:"
if command -v conda &> /dev/null; then
    echo "   conda activate mor_env"
else
    echo "   source mor_env/bin/activate"
fi
echo "2. 运行程序: python MOR.py"
echo ""
echo "注意事项："
echo "- 某些依赖可能需要系统级安装（如MPI）"
echo "- 如果遇到安装问题，请参考README.md中的详细说明"
echo "- 建议在Linux环境下运行以获得最佳性能" 