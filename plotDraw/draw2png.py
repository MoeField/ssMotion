# 绘制采集的数据到折线图

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

from pathlib import Path
matplotlib.use('TkAgg')
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
    # 击球类型列表   
    motions=[
        '正发','反发',                        #发球
        '杀1-高-吊', '杀-高2-吊', '杀-高-吊3',  #杀球
        '正挑','反挑',  #挑球
        '正抽','反抽',  #平抽球
        '推球'           #推球
    ]

    try:
        pltShower = mplDataShower()
        
        for motion in motions:
            files = pltShower.listAllFiles(f"trainData/{motion}")
            print(f"motion: {motion}")
            
            for csvfile in files:
                print(f"csvfile: {csvfile}")
                if csvfile.endswith(".csv"):
                    pltShower.showData(csvfile, _block=False)
                    plt.savefig(csvfile.replace(".csv", ".png"), dpi=300, transparent=True)
                    #plt.close()
    except Exception as e:
        print(f"Error occurred: {e}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Exiting the program.")
    exit(0)
