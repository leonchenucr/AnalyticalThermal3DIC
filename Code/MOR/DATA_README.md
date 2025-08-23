# 数据文件说明

## 大文件位置

由于GitHub有100MB文件大小限制，以下大文件未包含在仓库中：

### 必需的数据文件
- `3.90X90xihua.txt` (1.1GB) - COMSOL参考数据（主要）
- `3.90X90.txt` (137MB) - COMSOL参考数据（备用）
- `comsol_s8_pd_1_400x400.txt` (3.8MB) - 功率密度数据

### 获取方法

#### 方法1：从原始位置复制
```bash
# 如果你有原始文件，请复制到MOR文件夹
cp /path/to/original/3.90X90xihua.txt Code/MOR/
cp /path/to/original/3.90X90.txt Code/MOR/
cp /path/to/original/comsol_s8_pd_1_400x400.txt Code/MOR/
```

#### 方法2：下载链接
如果这些文件在其他地方可用，请下载并放置到MOR文件夹中。

#### 方法3：联系作者
请联系项目维护者获取这些数据文件。

## 文件用途

- **3.90X90xihua.txt**: 主要的COMSOL参考温度数据，用于验证MOR计算精度
- **3.90X90.txt**: 备用的COMSOL参考数据
- **comsol_s8_pd_1_400x400.txt**: 芯片功率密度分布数据，用于热源建模

## 注意事项

- 这些文件对于运行完整的MOR分析是必需的
- 没有这些文件，程序将无法进行COMSOL对比验证
- 建议使用SSD存储以提高大文件读取性能

## 替代方案

如果无法获取这些大文件，可以：
1. 使用模拟数据运行基本功能
2. 修改代码跳过COMSOL对比部分
3. 使用较小的测试数据集 