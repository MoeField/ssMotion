import plistlib
import threading
from datetime import datetime
import time
import pandas as pd
import numpy as np

from pathlib import Path

#from pyqtgraph.parametertree.parameterTypes import file
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataTypes.roundBuffer import get_dataBuffer
from dataTypes.ptData import get_ptData ,PtData
from config.conf import *


class DataStore:
    def __init__(self, imAddr:str='test', FdrName:str="trainData/test"):
        self.imAddr = imAddr
        self.dataRoundBuffer = get_dataBuffer(imAddr)
        self.ptData = get_ptData(imAddr)

        self.FdrName = FdrName
        folder_path = Path(self.FdrName)  # 替换为你的文件夹路径
        if not folder_path.exists():
            folder_path.mkdir(parents=True, exist_ok=True)  # parents=True 可以创建多级目录
            print(f"文件夹 '{folder_path}' 已创建。")
        else:
            print(f"文件夹 '{folder_path}' 已存在。")

        
        self.dataRoundBuffer = get_dataBuffer(imAddr)
        self.lastsaved:str= ""
        self.whenToSaveMotion = np.array([],dtype = np.int8)
        """
        当检测到超出阈值时，append MotionPreSize，之后每收到一帧数据就-1，直到为0时保存数据
        self.whenToSaveMotion-=1
        self.whenToSaveMotion=np.append(self.whenToSaveMotion, MotionPreSize)
        np.delete(self.whenToSaveMotion, np.where(self.whenToSaveMotion == 0)) 
        """

    def checkIfDataNeedToSave(self,framesPassed=UPDATE_EVERY_N_FRAMES):
        self.whenToSaveMotion-=framesPassed
        #save if motion clip Ok
        isZeroInWhenToSaveMotion = np.where(self.whenToSaveMotion <= 0) # 找到为0的元素
        if len(isZeroInWhenToSaveMotion[0])>0: # 有元素为0，保存数据
            #print(f"save data {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.saveData(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            self.whenToSaveMotion=np.delete(self.whenToSaveMotion, isZeroInWhenToSaveMotion) # 删除为0的元素

        dataBuffer = self.dataRoundBuffer.getAll()
        # 检查是否需要保存数据        #取最新framesPassed帧数据
        dataAcc = np.concatenate((
            dataBuffer["acc"]['X'][-framesPassed:],
            dataBuffer["acc"]['Y'][-framesPassed:],
            dataBuffer["acc"]['Z'][-framesPassed:]
        ))
        dataGryo = np.concatenate((
            dataBuffer["gyro"]['X'][-framesPassed:],
            dataBuffer["gyro"]['Y'][-framesPassed:],
            dataBuffer["gyro"]['Z'][-framesPassed:]
        ))

        # 检测是否超出阈值(触发保存动作)
        if np.max(np.abs(dataAcc)) > ACC_Threshold:    # by acc
            self.whenToSaveMotion=np.append(self.whenToSaveMotion, MotionPreSize)
        elif np.max(np.abs(dataGryo)) > GYRO_Threshold:    # by gyro
            self.whenToSaveMotion=np.append(self.whenToSaveMotion, MotionPreSize)
        else:
            pass
        
    def saveData(self, filename):
        if filename == self.lastsaved: # 避免重复保存
            return
        self.lastsaved = filename
        dataBuffer = self.dataRoundBuffer.getAll()
        df = pd.DataFrame({
            "timeStamp": dataBuffer["timeStamp"],
            "accX": dataBuffer["acc"]['X'],
            "accY": dataBuffer["acc"]['Y'],
            "accZ": dataBuffer["acc"]['Z'],
            "gyroX": dataBuffer["gyro"]['X'],
            "gyroY": dataBuffer["gyro"]['Y'],
            "gyroZ": dataBuffer["gyro"]['Z'],
            "angleX": dataBuffer["angle"]['X'],
            "angleY": dataBuffer["angle"]['Y'],
            "height": dataBuffer["height"]
        })
        self.ptData.update(df) # 更新ptData
        df.to_csv(f"./{self.FdrName}/{filename}.csv", index=False)
        


######################################################################
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
class mplDataShower:
    """
    输入csv文件路径，显示数据
    输入为pandas.DataFrame
    输出为matplotlib.pyplot.figure
    """
    def __init__(self):
        self.fig = None
        self.ax = None
        self.lines = {}
        self.plot_info = [
            (["accX", "accY", "accZ"], "acc"),
            (["gyroX", "gyroY", "gyroZ"], "gyro"),
            (["angleX", "angleY"], "angle"),
            (["height"], "height")
        ]
        self._init_plot_elements()
        #self.showData()

    def _init_plot_elements(self):
        # 完全重建图形对象
        if self.fig:
            plt.close(self.fig)
        
        self.fig, self.ax = plt.subplots(2, 2, figsize=(15,8), sharex=True)
        self.ax = self.ax.flatten()
        plt.subplots_adjust(wspace=0.3)
        self.lines = {}
        
        colors = ["red", "green", "blue"]
        for i, (columns, title) in enumerate(self.plot_info):
            self.ax[i].clear()
            self.ax[i].set_title(title)
            self.ax[i].set_xlabel("timeStamp")
            self.ax[i].set_ylabel(title)
            for j, col in enumerate(columns):
                line, = self.ax[i].plot([], [], label=col, color=colors[j%3])  # 创建空曲线
                self.lines[(i,col)] = line
            self.ax[i].legend()
        plt.gcf().canvas.manager.set_window_title("init")

    def showData(self, csvName:str=None,_block=False):
        # 每次显示前强制重建图形
        self._init_plot_elements()
        
        df=None
        if csvName:
            df = pd.read_csv(csvName)
        else: #图表归零
            print("no csvName, return")
            return self.fig

        # 更新各曲线数据
        for (i, col), line in self.lines.items():
            if col in df.columns:
                line.set_data(df["timeStamp"], df[col])
        
        # 自动调整坐标轴范围
        for ax in self.ax:
            ax.relim()
            ax.autoscale_view()
        
        # 刷新显示
        plt.gcf().canvas.manager.set_window_title(csvName)
        plt.draw()
        plt.pause(0.001)
        # 定义颜色列表
        colors = ["red", "green", "blue"]
        
        for i, (columns, title) in enumerate(self.plot_info):
            for j, col in enumerate(columns):
                self.ax[i].plot(df["timeStamp"], df[col], color=colors[j % len(colors)])
            self.ax[i].legend()
            self.ax[i].set_title(title)
            self.ax[i].set_xlabel("timeStamp")
            self.ax[i].set_ylabel(title)
        self.fig.canvas.draw_idle()
        plt.show(block=_block)
        plt.pause(0.1)  # 延长暂停时间确保窗口初始化

        return self.fig
    
    def listAllFiles(self, fdrName)->list[str]:
        folder_path = Path(fdrName)  # 替换为你的文件夹路径
        if not folder_path.exists():
            return []
        else:
            return [str(file) for file in folder_path.iterdir() if file.is_file()] # return lis


if __name__ == "__main__":
    pltShower = mplDataShower()
    
    motions=[
        '正发','反发',  #发球
        '杀球', '高远', '吊球', 
        '正挑','反挑',  #挑球
        '正抽','反抽',  #平抽球
        '推球'           #推球
    ]

    for motion in motions:
        files = pltShower.listAllFiles(f"trainData/{motion}")
        print(f"motion: {motion}")
        
        for csvfile in files:
            if csvfile.endswith(".csv"):
                pltShower.showData(csvfile,_block=True)
                #plt.savefig(csvfile.replace(".csv", ".png"), dpi=300)
            #time.sleep(0.5)
            #plt.close()