"""
integrated_main_gui_v2.py - 3D IC热仿真分析GUI系统（简化版）
使用 thermal_solver_3d.py 替代 3d_11layer.py

使用此版本需要:
1. 将 3d_11layer.py 重命名为 thermal_solver_3d.py
2. 或者直接使用我提供的 thermal_solver_3d.py 文件
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                             QFileDialog, QTextEdit, QGroupBox, QGridLayout,
                             QProgressBar, QTabWidget, QSpinBox, QDoubleSpinBox,
                             QMessageBox, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QTextCursor
from datetime import datetime


class SimulationWorker(QThread):
    """后台仿真线程 - 依次调用5个独立模块"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_running = True
        
    def run(self):
        try:
            # 动态导入模块
            self.progress.emit(1, "正在加载核心模块...")
            
            try:
                from curve_recognition import extract_contours
                from rectangle_filling import generate_rectangles
                from rectangular_corresponding_power import calculate_power
                from mesh_generation import generate_mesh
                from thermal_solver_3d import solve_thermal  # 使用重命名后的模块
                
                self.progress.emit(3, "✓ 所有模块加载成功")
                
            except ImportError as e:
                error_details = str(e)
                if "thermal_solver_3d" in error_details:
                    raise RuntimeError(
                        f"无法导入热求解模块!\n\n"
                        f"请确保以下文件存在:\n"
                        f"✓ curve_recognition.py\n"
                        f"✓ rectangle_filling.py\n"
                        f"✓ rectangular_corresponding_power.py\n"
                        f"✓ mesh_generation.py\n"
                        f"✗ thermal_solver_3d.py (缺失!)\n\n"
                        f"解决方案:\n"
                        f"1. 将 3d_11layer.py 重命名为 thermal_solver_3d.py\n"
                        f"2. 或使用我提供的 thermal_solver_3d.py 文件"
                    )
                else:
                    raise RuntimeError(
                        f"模块导入失败: {error_details}\n\n"
                        f"请确保所有5个模块文件在同一目录:\n"
                        f"- curve_recognition.py\n"
                        f"- rectangle_filling.py\n"
                        f"- rectangular_corresponding_power.py\n"
                        f"- mesh_generation.py\n"
                        f"- thermal_solver_3d.py"
                    )
            
            output_dir = self.config['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            # 定义中间文件路径
            curve_file = os.path.join(output_dir, 's8_curve_points1.txt')
            funit_file = os.path.join(output_dir, 's8_FUnit1.txt')
            power_file = os.path.join(output_dir, 's8_power1.txt')
            mesh_file = os.path.join(output_dir, 's8_pd_test_S.txt')
            
            # ===== 步骤1: 轮廓识别 (0-20%) =====
            if not self._is_running:
                return
            self.progress.emit(5, "正在读取芯片布局图像...")
            self.progress.emit(10, "执行边缘检测和轮廓提取...")
            
            extract_contours(
                self.config['image_path'],
                curve_file,
                self.config['domain_width'],
                self.config['domain_height']
            )
            self.progress.emit(20, "✓ 轮廓识别完成")
            
            # ===== 步骤2: 矩形生成 (20-40%) =====
            if not self._is_running:
                return
            self.progress.emit(25, "生成矩形热源区域...")
            
            generate_rectangles(
                curve_file,
                funit_file,
                self.config['domain_width'],
                self.config['domain_height']
            )
            self.progress.emit(40, "✓ 矩形区域生成完成")
            
            # ===== 步骤3: 功率映射 (40-55%) =====
            if not self._is_running:
                return
            self.progress.emit(45, "计算功率分布...")
            
            calculate_power(
                funit_file,
                power_file,
                self.config['power_coefficient']
            )
            self.progress.emit(55, "✓ 功率分布计算完成")
            
            # ===== 步骤4: 网格生成 (55-70%) =====
            if not self._is_running:
                return
            self.progress.emit(60, "生成共形网格...")
            
            generate_mesh(
                power_file,
                funit_file,
                mesh_file,
                self.config['domain_width'],
                self.config['domain_height']
            )
            self.progress.emit(70, "✓ 网格生成完成")
            
            # ===== 步骤5: 3D热分析 (70-95%) =====
            if not self._is_running:
                return
            self.progress.emit(75, "初始化热求解器...")
            self.progress.emit(80, "求解线性系统...")
            
            thermal_results = solve_thermal(
                mesh_file=mesh_file,
                boundary_file=self.config.get('boundary_path', None) if self.config.get('boundary_path') else None,
                comsol_file=self.config.get('comsol_path', None) if self.config.get('comsol_path') else None,
                output_dir=output_dir,
                num_eigen=self.config['num_eigen']
            )
            self.progress.emit(95, "✓ 热分析完成")
            
            # ===== 步骤6: 整理结果 =====
            if not self._is_running:
                return
            self.progress.emit(98, "生成可视化结果...")
            
            results = {
                'mean_error': thermal_results.get('mean_error', 0),
                'max_error': thermal_results.get('max_error', 0),
                'max_temp': thermal_results.get('max_temp', 0),
                'min_temp': thermal_results.get('min_temp', 0),
                'calc_time': thermal_results.get('calc_time', 0),
                'output_dir': output_dir
            }
            
            self.progress.emit(100, "🎉 仿真成功完成!")
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            error_msg = f"仿真错误: {str(e)}\n\n详细信息:\n{traceback.format_exc()}"
            self.error.emit(error_msg)
    
    def stop(self):
        """停止运行"""
        self._is_running = False


class ThermalSimGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D IC 热仿真分析系统 v2.1 - 简化版")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
            }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # 创建左右分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧控制面板
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # 右侧结果面板
        right_panel = self.create_result_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
    def create_control_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        title = QLabel("🔥 3D IC 热仿真分析")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # 输入文件组
        input_group = self.create_input_group()
        layout.addWidget(input_group)
        
        # 参数配置组
        param_group = self.create_parameter_group()
        layout.addWidget(param_group)
        
        # 输出目录
        output_group = self.create_output_group()
        layout.addWidget(output_group)
        
        # 控制按钮
        control_group = self.create_control_buttons()
        layout.addWidget(control_group)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 日志区域
        log_group = self.create_log_group()
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        return panel
    
    def create_input_group(self):
        """创建输入文件组"""
        group = QGroupBox("📁 输入文件")
        layout = QGridLayout()
        
        # 图像文件
        layout.addWidget(QLabel("芯片布局图像:"), 0, 0)
        self.image_path = QLineEdit()
        self.image_path.setPlaceholderText("选择 PNG/JPG 图像文件...")
        layout.addWidget(self.image_path, 0, 1)
        btn_image = QPushButton("浏览")
        btn_image.clicked.connect(lambda: self.browse_file(self.image_path, "Image Files (*.png *.jpg)"))
        layout.addWidget(btn_image, 0, 2)
        
        # 边界条件
        layout.addWidget(QLabel("边界条件:"), 1, 0)
        self.boundary_path = QLineEdit()
        self.boundary_path.setPlaceholderText("选择边界条件文件(可选)...")
        layout.addWidget(self.boundary_path, 1, 1)
        btn_boundary = QPushButton("浏览")
        btn_boundary.clicked.connect(lambda: self.browse_file(self.boundary_path, "Text Files (*.txt)"))
        layout.addWidget(btn_boundary, 1, 2)
        
        # COMSOL数据
        layout.addWidget(QLabel("COMSOL验证:"), 2, 0)
        self.comsol_path = QLineEdit()
        self.comsol_path.setPlaceholderText("选择 COMSOL 数据文件(可选)...")
        layout.addWidget(self.comsol_path, 2, 1)
        btn_comsol = QPushButton("浏览")
        btn_comsol.clicked.connect(lambda: self.browse_file(self.comsol_path, "Text Files (*.txt)"))
        layout.addWidget(btn_comsol, 2, 2)
        
        group.setLayout(layout)
        return group
    
    def create_parameter_group(self):
        """创建参数配置组"""
        group = QGroupBox("⚙️ 仿真参数")
        layout = QGridLayout()
        
        # 功率密度系数
        layout.addWidget(QLabel("功率密度系数:"), 0, 0)
        self.power_coef = QDoubleSpinBox()
        self.power_coef.setDecimals(2)
        self.power_coef.setRange(0, 1e12)
        self.power_coef.setValue(9.81e9)
        self.power_coef.setSuffix(" W/m³")
        layout.addWidget(self.power_coef, 0, 1)
        
        # 特征模式数
        layout.addWidget(QLabel("特征模式数:"), 1, 0)
        self.num_eigen = QSpinBox()
        self.num_eigen.setRange(10, 100)
        self.num_eigen.setValue(30)
        layout.addWidget(self.num_eigen, 1, 1)
        
        # 域尺寸
        layout.addWidget(QLabel("域宽度 (m):"), 2, 0)
        self.domain_width = QDoubleSpinBox()
        self.domain_width.setDecimals(4)
        self.domain_width.setValue(0.024)
        layout.addWidget(self.domain_width, 2, 1)
        
        layout.addWidget(QLabel("域高度 (m):"), 3, 0)
        self.domain_height = QDoubleSpinBox()
        self.domain_height.setDecimals(4)
        self.domain_height.setValue(0.024)
        layout.addWidget(self.domain_height, 3, 1)
        
        group.setLayout(layout)
        return group
    
    def create_output_group(self):
        """创建输出目录组"""
        group = QGroupBox("💾 输出目录")
        layout = QHBoxLayout()
        
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("选择输出目录...")
        self.output_dir.setText("./outputs")
        layout.addWidget(self.output_dir)
        
        btn_output = QPushButton("选择")
        btn_output.clicked.connect(self.browse_output_dir)
        layout.addWidget(btn_output)
        
        group.setLayout(layout)
        return group
    
    def create_control_buttons(self):
        """创建控制按钮"""
        group = QGroupBox("🎮 控制")
        layout = QHBoxLayout()
        
        self.btn_start = QPushButton("▶️ 开始仿真")
        self.btn_start.clicked.connect(self.start_simulation)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("⏹️ 停止")
        self.btn_stop.clicked.connect(self.stop_simulation)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        layout.addWidget(self.btn_stop)
        
        group.setLayout(layout)
        return group
    
    def create_log_group(self):
        """创建日志组"""
        group = QGroupBox("📋 运行日志")
        layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
            }
        """)
        layout.addWidget(self.log_text)
        
        group.setLayout(layout)
        return group
    
    def create_result_panel(self):
        """创建右侧结果面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 结果统计
        stats_group = self.create_stats_group()
        layout.addWidget(stats_group)
        
        # 图像显示标签页
        self.result_tabs = QTabWidget()
        self.result_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #cccccc;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4CAF50;
                color: white;
            }
        """)
        
        # 创建四个标签页
        self.tab_preview = self.create_image_tab("预览")
        self.tab_calculated = self.create_image_tab("计算温度")
        self.tab_comsol = self.create_image_tab("COMSOL对比")
        self.tab_error = self.create_image_tab("误差分布")
        
        self.result_tabs.addTab(self.tab_preview, "📷 图像预览")
        self.result_tabs.addTab(self.tab_calculated, "🌡️ 计算结果")
        self.result_tabs.addTab(self.tab_comsol, "✅ COMSOL")
        self.result_tabs.addTab(self.tab_error, "📊 误差分析")
        
        layout.addWidget(self.result_tabs)
        
        return panel
    
    def create_stats_group(self):
        """创建统计信息组"""
        group = QGroupBox("📈 仿真结果统计")
        layout = QGridLayout()
        
        self.stat_labels = {}
        stats = [
            ("最高温度:", "max_temp", " K"),
            ("最低温度:", "min_temp", " K"),
            ("平均误差:", "mean_error", " K"),
            ("最大误差:", "max_error", " K"),
            ("计算时间:", "calc_time", " s")
        ]
        
        for i, (label, key, unit) in enumerate(stats):
            layout.addWidget(QLabel(label), i // 3, (i % 3) * 2)
            value_label = QLabel("--")
            value_label.setStyleSheet("font-weight: bold; color: #2196F3;")
            self.stat_labels[key] = (value_label, unit)
            layout.addWidget(value_label, i // 3, (i % 3) * 2 + 1)
        
        group.setLayout(layout)
        return group
    
    def create_image_tab(self, name):
        """创建图像显示标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label = QLabel(f"暂无 {name}数据")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 5px;
                padding: 20px;
                color: #999999;
                font-size: 14pt;
            }
        """)
        label.setMinimumHeight(400)
        layout.addWidget(label)
        
        return widget
    
    def browse_file(self, line_edit, file_filter):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", file_filter)
        if file_path:
            line_edit.setText(file_path)
            self.log(f"已选择文件: {os.path.basename(file_path)}")
            
            # 如果是图像,显示预览
            if file_filter.startswith("Image"):
                self.display_image_preview(file_path)
    
    def browse_output_dir(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir.setText(dir_path)
            self.log(f"输出目录: {dir_path}")
    
    def display_image_preview(self, image_path):
        """显示图像预览"""
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            layout = self.tab_preview.layout()
            old_label = layout.itemAt(0).widget()
            
            new_label = QLabel()
            new_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            new_label.setAlignment(Qt.AlignCenter)
            
            layout.replaceWidget(old_label, new_label)
            old_label.deleteLater()
    
    def start_simulation(self):
        """开始仿真"""
        # 验证输入
        if not self.image_path.text():
            QMessageBox.warning(self, "输入错误", "请选择芯片布局图像!")
            return
        
        # 创建输出目录
        output_dir = self.output_dir.text()
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备配置
        config = {
            'image_path': self.image_path.text(),
            'boundary_path': self.boundary_path.text(),
            'comsol_path': self.comsol_path.text(),
            'power_coefficient': self.power_coef.value(),
            'num_eigen': self.num_eigen.value(),
            'domain_width': self.domain_width.value(),
            'domain_height': self.domain_height.value(),
            'output_dir': output_dir
        }
        
        # 禁用开始按钮,启用停止按钮
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.log("=" * 50)
        self.log(f"开始仿真 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        self.log("=" * 50)
        
        # 启动后台线程
        self.worker = SimulationWorker(config)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.simulation_finished)
        self.worker.error.connect(self.simulation_error)
        self.worker.start()
    
    def stop_simulation(self):
        """停止仿真"""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.worker.terminate()
            self.log("⚠️ 仿真已被用户中止")
            self.reset_ui()
    
    def update_progress(self, value, message):
        """更新进度"""
        self.progress_bar.setValue(value)
        self.log(f"[{value}%] {message}")
    
    def simulation_finished(self, results):
        """仿真完成"""
        self.log("=" * 50)
        self.log("✅ 仿真成功完成!")
        self.log("=" * 50)
        
        # 更新统计数据
        for key, (label, unit) in self.stat_labels.items():
            if key in results:
                label.setText(f"{results[key]:.2f}{unit}")
        
        # 加载结果图像
        self.load_result_images(results['output_dir'])
        
        self.reset_ui()
        
        QMessageBox.information(self, "完成", "仿真计算成功完成!")
    
    def simulation_error(self, error_msg):
        """仿真错误"""
        self.log(f"❌ 错误: {error_msg}")
        self.reset_ui()
        QMessageBox.critical(self, "错误", f"仿真失败:\n{error_msg}")
    
    def load_result_images(self, output_dir):
        """加载结果图像"""
        images = {
            'calculated_temperature_mor_3d.png': self.tab_calculated,
            'comsol_temperature_3d.png': self.tab_comsol,
            'chip_layer_error_distribution.png': self.tab_error
        }
        
        for filename, tab in images.items():
            path = os.path.join(output_dir, filename)
            if os.path.exists(path):
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    layout = tab.layout()
                    old_label = layout.itemAt(0).widget()
                    
                    new_label = QLabel()
                    new_label.setPixmap(pixmap.scaled(900, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    new_label.setAlignment(Qt.AlignCenter)
                    new_label.setStyleSheet("background-color: white; padding: 10px;")
                    
                    layout.replaceWidget(old_label, new_label)
                    old_label.deleteLater()
    
    def reset_ui(self):
        """重置UI状态"""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.moveCursor(QTextCursor.End)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = ThermalSimGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()