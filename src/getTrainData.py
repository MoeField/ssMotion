import asyncio
import time

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from bluetooth.scan import fetchMacAddr
from bluetooth.receive import ImDataReceiver
from plotDraw.tdiPlotDrawer import PyQtGraphDrawer
from src.motionCutter import DataStore

MotionName = 'm_name'


if __name__ == "__main__":
    # 运行扫描函数
    imAddrs=[]
    asyncio.run(fetchMacAddr(toWhere=imAddrs))
    if len(imAddrs) == 0:
        print("No device found")
        exit()
    # 接收数据函数
    drc=ImDataReceiver(imAddrs[0])
    # 绘图函数
    dDrawer=PyQtGraphDrawer(imAddrs[0])
    # 数据存储函数
    ds=DataStore(imAddrs[0],FdrName=f"trainData/{MotionName}")

    drc.notiFxList.append(dDrawer.update)
    drc.notiFxList.append(ds.checkIfDataNeedToSave)

    dDrawer.run()
    drc.run()
    dDrawer.app.exec_()
