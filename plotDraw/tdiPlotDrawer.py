import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import threading
import time

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from dataTypes.roundBuffer import dataType_ImuFrame, RoundBuffer, get_dataBuffer
from dataTypes.runFlag import get_runFalg

TIMER_TIME_OUT = 20 # ms
AntiAlias = False
AUTO_UPDATE=False

class PyQtGraphDrawer:
    def __init__(self, imAddr:str):
        self.imAddr = imAddr
        self.dataBuffer = get_dataBuffer(imAddr)
        self.rFlg = get_runFalg(imAddr)
        self.rFlg.set(False) 
        self.colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']  # 颜色列表
        self.mtCNames = ['X', 'Y', 'Z', 'W']  # 曲线名称列表
    
    def run(self):
        self.rFlg.set(True)
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title="传感器数据监测")
        self.win.resize(1500, 900)
        self.win.setBackground('w')  # 设置主窗口背景为白色
        self.win.closeEvent = self.closeEvent

        # 子图布局配置
        # 初始化子图并设置X轴联动
        self.plots = {
            'acc': self.win.addPlot(title='加速度计', row=0, col=0),
            'ACC': self.win.addPlot(title='线性加速度', row=0, col=1),
            'acs': self.win.addPlot(title='角速度', row=0, col=2),
            'angle': self.win.addPlot(title='欧拉角', row=1, col=0),
            'gyro': self.win.addPlot(title='陀螺仪', row=1, col=1),
            'mag': self.win.addPlot(title='磁力计', row=1, col=2),
            'quat': self.win.addPlot(title='四元数', row=2, col=0),
            'offset': self.win.addPlot(title='偏移量', row=2, col=1),
            'height': self.win.addPlot(title='高度', row=2, col=2),
            'temp': self.win.addPlot(title='温度', row=3, col=0),
            'press': self.win.addPlot(title='气压', row=3, col=1),
            'voltage': self.win.addPlot(title='电压', row=3, col=2)
        }

        # 设置行级X轴联动
        subPltNames = [
            'acc', 'ACC', 'acs',
            'angle', 'gyro', 'mag',
            'quat', 'offset', 'height',
            'temp', 'press'#, 'voltage'
        ]

        main_plot = self.plots['voltage']
        main_plot.setLabel('bottom', '时间', units='ms') # 设置主X轴标签
        for subplt in subPltNames:
            self.plots[subplt].setXLink(main_plot)
        
        # 曲线配置
        self.curves = {}
        
        for name, plot in self.plots.items():
            plot.showGrid(x=True, y=True)
            plot.setLabel('left', '值')
            
            _disableAutoRange = True
            # 设置固定Y轴范围
            if name == 'angle':     plot.setYRange(-180, 180)
            elif name in ['acc', 'ACC', 'acs']: plot.setYRange(-120, 120)
            elif name == 'gyro':    plot.setYRange(-1500, 1500)
            elif name == 'mag':     plot.setYRange(-50, 50)
            elif name == 'quat':    plot.setYRange(-1, 1)
            elif name == 'offset':  plot.setYRange(-25, 25)
            else: _disableAutoRange = False
    
            plot.setMouseEnabled(x=False, y=False)  # 禁用鼠标交互
            if _disableAutoRange:
                plot.setAutoVisible(y=False)    # 禁用Y轴自动可见性调整
                plot.setAutoVisible(y=False)    # 禁用Y轴自动可见性调整

            if name == 'quat':
                self.curves[name] = [plot.plot(pen=pg.mkPen(c, width=2)) for c in self.colors]
            elif name in ['temp','press','height','voltage']:
                self.curves[name] = [plot.plot(pen=pg.mkPen('#888888', width=2))]
            else:
                self.curves[name] = [plot.plot(pen=pg.mkPen(c, width=2)) for c in self.colors[:3]]
        
        # 定时器
        if AUTO_UPDATE:
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update)
            self.timer.start(TIMER_TIME_OUT)    # 定时器触发间隔
        else:
            self.timer = None

    def closeEvent(self, event):
        print("closeEvent @PyQtGraphDrawer")
        self.rFlg.set(False)
        self.app.quit()
        event.accept()

    # 在PyQtGraphDrawer类中更新数据访问方式
    def update(self):
        # 获取整个缓冲区数据
        buffer_data = self.dataBuffer.getAll()
        # 时间戳
        ts = buffer_data['timeStamp']
        if 0 in ts:
            return
        min_ts = ts.min()
        max_ts = ts.max()
        # 动态设置主图X轴范围（自动同步其他子图）
        #for plot in self.plots.values(): plot.setXRange(min_ts, max_ts, padding=0) # 动态设置X轴范围
        self.plots['voltage'].setXRange(min_ts, max_ts, padding=0)
        # 更新各曲线数据
        for name, curve_list in self.curves.items():
            # 直接从结构化数组中获取字段数据
            data = buffer_data[name]
            if name in ['temp','press','height','voltage']:
                y = data
                curve_list[0].setData(ts, y, antialias=AntiAlias)
            else:
                for i, curve in enumerate(curve_list):
                    y = data[self.mtCNames[i]]
                    curve.setData(ts, y, antialias=AntiAlias)
    
    def __del__(self):
        #self.win.close()
        pass


if __name__ == "__main__":
    import random
    AUTO_UPDATE = True
    dataBuffer = get_dataBuffer('test')
    
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

    threading.Thread(target=data_generator, daemon=True).start()
    time.sleep(1)
    drawer = PyQtGraphDrawer('test')
    print("start")
    time.sleep(1)
    drawer.run()
    drawer.app.exec_()