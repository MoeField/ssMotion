import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from dataTypes.roundBuffer import dataType_ImuFrame, RoundBuffer, get_dataBuffer
from dataTypes.runFlag import get_runFalg
from dataTypes.ptData import get_ptData ,PtData

from pyqtgraph.Qt import QT_LIB
print(f"Using PyQtGraph with {QT_LIB}")

TIMER_TIME_OUT = 50 # ms
AntiAlias = False
AUTO_UPDATE = True

def calculate_peak(acc_data):
    """计算加速度矢量模量峰值"""
    acc3D = np.column_stack((acc_data['X'],acc_data['Y'],acc_data['Z']))
    acc_norm = np.linalg.norm(acc3D, axis=1)  # 输入形状：(80,3)
    return np.max(acc_norm)

def calculate_gyro_integral(gyro_y_data, fs=60):
    """计算Y轴角速度积分（采样率60Hz）"""
    time_step = 1 / fs
    return np.trapezoid(np.abs(gyro_y_data), dx=time_step)




class PyQtGraphDrawer:
    def __init__(self, imAddr:str):
        self.imAddr = imAddr
        self.dataBuffer = get_dataBuffer(imAddr)
        self.rFlg = get_runFalg(imAddr)
        self.rFlg.set(False)
        self.colors = ['#FF0000', '#00FF00', '#0000FF']
        self.ptData = get_ptData(imAddr)

        # Conditional registration based on Qt binding
        """
        if QT_LIB == 'PyQt5':
            QtCore.qRegisterMetaType('QVector<int>')
        elif QT_LIB == 'PySide':
            QtCore.QMetaType.registerType('QVector<int>')
        """

    def run(self):
        self.rFlg.set(True)
        # Removed problematic registration call
        self.app = QtWidgets.QApplication([])
        # Create main window with proper layout hierarchy
        self.win = QtWidgets.QMainWindow()
        self.win.closeEvent = self.closeEvent  # 注册关闭事件处理函数
        central_widget = QtWidgets.QWidget()
        self.win.setCentralWidget(central_widget)

        self.win.resize(1400, 600)

        self.win.setWindowTitle(f"IMU数据可视化 - {self.imAddr}")
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        # Create graphics layout container
        graphics_widget = pg.GraphicsLayoutWidget()
        #backgroundColor
        graphics_widget.setBackground('w')

        main_layout.addWidget(graphics_widget, 3)

        # 左侧2x2图表布局
        # Add graphics layout to main container
        # Access graphics layout from container
        graph_layout = graphics_widget.ci

        self.plots = {
            'acc': graph_layout.addPlot(title='加速度', row=0, col=0),
            'gyro': graph_layout.addPlot(title='陀螺仪', row=0, col=1),
            'angle': graph_layout.addPlot(title='角度', row=1, col=0),
            'height': graph_layout.addPlot(title='高度', row=1, col=1)
        }
        

        # 图表曲线初始化
        self.curves = {
            'acc': [self.plots['acc'].plot(pen=c) for c in self.colors],
            'gyro': [self.plots['gyro'].plot(pen=c) for c in self.colors],
            'angle': [self.plots['angle'].plot(pen=c) for c in self.colors],
            'height': self.plots['height'].plot(pen='#FFA500')
        }

        # 设置行级X轴联动
        main_plot = self.plots['height']
        for name, plot in self.plots.items():
            if name != 'height':
                plot.setXLink(main_plot)

        # 右侧表格布局
        # Create proxy container for Qt widgets
        # Create proxy container for tables
        table_widget = QtWidgets.QWidget()
        right_layout = QVBoxLayout(table_widget)
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(table_widget)
        # Add tables to main layout
        main_layout.addWidget(table_widget, 1)
        
        # 4x9表格
        self.stats_table = QTableWidget(9, 4)
        self.stats_table.setHorizontalHeaderLabels(['All','X', 'Y', 'Z'])
        self.stats_table.setVerticalHeaderLabels([
            'Acc Extreme',
            'Gyro Extreme',
            'Angle Extreme',
            'Height Extreme',

        ])
        right_layout.addWidget(self.stats_table, 2)

        # 识别结果表格
        self.prob_table = QTableWidget(2, 2)
        self.prob_table.setVerticalHeaderLabels(['SVM', 'CNN-LSTM', ''])
        self.prob_table.setHorizontalHeaderLabels(['Predict', 'Confidence', ''])
        # 设置表格列宽自适应
        self.stats_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.prob_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        right_layout.addWidget(self.prob_table, 1)

        # Add table widget to main container
        right_container = QtWidgets.QWidget()
        right_container.setLayout(right_layout)
        main_layout.addWidget(right_container, 1)

        # 显示主窗口
        self.win.show()

        # 定时器初始化
        # 定时器
        if AUTO_UPDATE:
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_all)
            self.timer.start(TIMER_TIME_OUT)    # 定时器触发间隔
        else:
            self.timer = None

    def update_all(self):
        # 检测结果表格更新
        if self.ptData.getData().empty:
            return

        svm_pred, svm_cfd = self.ptData.getSvmResult()
        cnn_pred, cnn_cfd = self.ptData.getCnnLstmResult()

        self.prob_table.setItem(0, 0, QTableWidgetItem(svm_pred))
        self.prob_table.setItem(0, 1, QTableWidgetItem(f"{svm_cfd:.2f}"))

        self.prob_table.setItem(1, 0, QTableWidgetItem(cnn_pred))
        self.prob_table.setItem(1, 1, QTableWidgetItem(f"{cnn_cfd:.2f}"))

        ###############################################################
        if self.ptData.times4get > 5:
            self.ptData.clear()
            return  #skip update graph

        buffer_data = self.dataBuffer.getAll()
        ts = buffer_data['timeStamp']
        
        # 统一设置主图表X轴范围
        if len(ts) > 0:
            min_ts = ts.min()
            max_ts = ts.max()
            self.plots['height'].setXRange(min_ts, max_ts, padding=0)
        
        # 更新图表数据
        for name in ['acc', 'gyro', 'angle', 'height']:
            data = buffer_data[name]
            if name == 'height':
                self.curves[name].setData(ts, data)
            else:
                for i, curve in enumerate(self.curves[name]):
                    curve.setData(ts, data[['X','Y','Z'][i]])

        # 统计表格更新
        for v, items in enumerate(['acc', 'gyro', 'angle']):
            for i, axis in enumerate(['X', 'Y', 'Z']):
                self.stats_table.setItem(v, i+1, QTableWidgetItem(
                    f"{np.max(buffer_data[items][axis]-np.min(buffer_data[items][axis])):.2f}"
                ))
        self.stats_table.setItem(0, 0, QTableWidgetItem(f"{calculate_peak(buffer_data['acc']):.2f}"))
        self.stats_table.setItem(1, 0, QTableWidgetItem(f"{calculate_peak(buffer_data['gyro']):.2f}"))
        self.stats_table.setItem(2, 0, QTableWidgetItem(f"{calculate_peak(buffer_data['angle']):.2f}"))
        #height
        self.stats_table.setItem(3, 0, QTableWidgetItem(f"{np.max(buffer_data['height']-np.min(buffer_data['height'])):.2f}"))


    def closeEvent(self, event):  # 显式绑定关闭事件
        print(f"PyQtGraphDrawer {self.imAddr} closing")
        self.rFlg.set(False)
        if self.timer is not None:
            self.timer.stop()
            self.timer.deleteLater()
            self.timer = None
        self.app.quit()
        event.accept()
        print(f"PyQtGraphDrawer {self.imAddr} closed")
        

if __name__ == '__main__':
    import threading
    import time
    AUTO_UPDATE = True
    dataBuffer = get_dataBuffer('test')
    drawer = PyQtGraphDrawer("test")
    
    # 在数据生成器线程中更新数据推送方式
    def data_generator():
        counter = 0
        while True:
            # 创建符合新数据结构的numpy数组
            tframeData = np.array((
                np.uint16(1),
                np.uint32(counter),
                (np.random.uniform(-15,15),np.random.uniform(-15,15),np.random.uniform(-15,15)),
                (np.random.uniform(-2,2),np.random.uniform(-2,2),np.random.uniform(-2,2)),
                (np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)),
                (np.random.uniform(-100,100),np.random.uniform(-100,100),np.random.uniform(-100,100)),
                np.float32(np.random.uniform(20,40)),
                np.float32(np.random.uniform(900,1100)),
                np.float32(np.random.uniform(0,100)),
                (np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)),
                (np.random.uniform(-180,180),np.random.uniform(-180,180),np.random.uniform(-180,180)),
                (np.random.uniform(-100,100),np.random.uniform(-100,100),np.random.uniform(-100,100)),
                True,False,True,False,
                (np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)),
                np.float32(np.random.uniform(3,4.2)),
                np.uint8(1)
            ), dtype=dataType_ImuFrame)
            counter += 1
            # 使用新的push方法
            dataBuffer.push(tframeData)
            time.sleep(0.05)
            #drawer.update_all()
    
    drawer.run()
    threading.Thread(target=data_generator, daemon=True).start()

    
    sys.exit(drawer.app.exec_())
