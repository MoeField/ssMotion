
import sys
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem,
                            QGridLayout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.tDrameData = pd.DataFrame()
        self.timer = QTimer()
        self.timer.setInterval(1000)  # 20ms间隔
        self.timer.timeout.connect(self.update_handler)
        self.timer.start()
    
    def update_handler(self):
        if not self.tDrameData.empty:
            self.update_plot()
            print("update")
        self.timer.setInterval(1000)  # 20ms间隔
        
    def initUI(self):
        self.setWindowTitle('PyQt示例')
        self.setGeometry(100, 100, 1200, 600)

        # 主布局（左右分区）
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 左侧绘图区域改用GridLayout
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget, 3)
        
        # 创建2x2子图
        self.plots = {
            'acc': self.plot_widget.addPlot(title='加速度', row=0, col=0),
            'gyro': self.plot_widget.addPlot(title='陀螺仪', row=0, col=1),
            'angle': self.plot_widget.addPlot(title='角度', row=1, col=0),
            'height': self.plot_widget.addPlot(title='高度', row=1, col=1)
        }
        
        # 初始化曲线对象
        self.curves = {
            'acc': [self.plots['acc'].plot(pen='r'), 
                    self.plots['acc'].plot(pen='g'),
                    self.plots['acc'].plot(pen='b')],
            'gyro': [self.plots['gyro'].plot(pen='r'),
                     self.plots['gyro'].plot(pen='g'),
                     self.plots['gyro'].plot(pen='b')],
            'angle': [self.plots['angle'].plot(pen='r'),
                      self.plots['angle'].plot(pen='g')],
            'height': self.plots['height'].plot(pen='y')
        }

        # 右侧控制面板
        right_panel = QVBoxLayout()
        
        # 数据表格
        self.table1 = QTableWidget(5, 3)
        self.table1.setHorizontalHeaderLabels(['列1', '列2', '列3'])
        self.table1.setItem(0, 0, QTableWidgetItem('示例数据'))
        right_panel.addWidget(self.table1)

        self.table2 = QTableWidget(3, 2)
        self.table2.setHorizontalHeaderLabels(['A', 'B'])
        right_panel.addWidget(self.table2)

        # 功能按钮
        btn1 = QPushButton('按钮1')
        #btn1.clicked.connect(lambda: print('hi'))
        right_panel.addWidget(btn1)

        btn2 = QPushButton('按钮2')
        #btn2.clicked.connect(lambda: print('bye'))
        right_panel.addWidget(btn2)

        update_btn = QPushButton('更新图表')
        #update_btn.clicked.connect(self.update_plot)
        right_panel.addWidget(update_btn)

        layout.addLayout(right_panel, 1)

    def update_plot(self):
        df = self.tDrameData.copy()
        # 从DataFrame提取数据
        if df.empty: 
            raise ValueError("DataFrame is empty")
            return
        # 更新加速度曲线
        for i, axis in enumerate(['X','Y','Z']):
            self.curves['acc'][i].setData(df['timeStamp'], df[f'acc{axis}'])
        
        # 更新陀螺仪曲线
        for i, axis in enumerate(['X','Y','Z']):
            self.curves['gyro'][i].setData(df['timeStamp'], df[f'gyro{axis}'])
        
        # 更新角度曲线
        for i, axis in enumerate(['X','Y']):
            self.curves['angle'][i].setData(df['timeStamp'], df[f'angle{axis}'])
        
        # 更新高度曲线
        self.curves['height'].setData(df['timeStamp'], df['height'])
        """
        # 更新表格1
        self.table1.setItem(0, 0, QTableWidgetItem(f"{df['accX'].mean():.2f}"))
        self.table1.setItem(0, 1, QTableWidgetItem(f"{df['accY'].mean():.2f}"))
        self.table1.setItem(0, 2, QTableWidgetItem(f"{df['accZ'].mean():.2f}"))
        self.table1.setItem(1, 0, QTableWidgetItem(f"{df['gyroX'].mean():.2f}"))
        self.table1.setItem(1, 1, QTableWidgetItem(f"{df['gyroY'].mean():.2f}"))
        self.table1.setItem(1, 2, QTableWidgetItem(f"{df['gyroZ'].mean():.2f}"))
        self.table1.setItem(2, 0, QTableWidgetItem(f"{df['angleX'].mean():.2f}"))
        self.table1.setItem(2, 1, QTableWidgetItem(f"{df['angleY'].mean():.2f}"))
        self.table1.setItem(2, 2, QTableWidgetItem(f"{df['height'].mean():.2f}"))
        # 更新表格2
        self.table2.setItem(0, 0, QTableWidgetItem(f"{df['accX'].std():.2f}"))
        self.table2.setItem(0, 1, QTableWidgetItem(f"{df['accY'].std():.2f}"))
        self.table2.setItem(1, 0, QTableWidgetItem(f"{df['gyroX'].std():.2f}"))
        self.table2.setItem(1, 1, QTableWidgetItem(f"{df['gyroY'].std():.2f}"))
        self.table2.setItem(2, 0, QTableWidgetItem(f"{df['angleX'].std():.2f}"))
        self.table2.setItem(2, 1, QTableWidgetItem(f"{df['angleY'].std():.2f}"))
        """
        self.plot_widget.update()
        self.plot_widget.show()


    def update_now(self):
        print("update now")
        self.timer.setInterval(10)  # 20ms间隔

app = QApplication(sys.argv)
main = MainWindow()
import pathlib
import random
if __name__ == '__main__':
    main.show()
    """    
    #all data in testData dir
    for file in pathlib.Path("testData").iterdir():
        if file.is_file() and file.suffix == ".csv":
            print(f"loading {file}")
            main.tDrameData = pd.read_csv(file)
            main.update_now()
            
            """

    sys.exit(app.exec_())