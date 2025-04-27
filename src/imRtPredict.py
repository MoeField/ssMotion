import asyncio
import time

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

#from dataTypes.roundBuffer import dataType_ImuFrame, RoundBuffer, get_dataBuffer
from dataTypes.runFlag import get_runFalg
from bluetooth.scan import fetchMacAddr
from bluetooth.receive import ImDataReceiver
from plotDraw.rcgPlotDrawer import PyQtGraphDrawer
from src.motionCutter import DataStore

MotionName = 'testGame1'

if __name__ == "__main__":
    # 运行扫描函数
    imAddrs=[]
    asyncio.run(fetchMacAddr(toWhere=imAddrs))
    if len(imAddrs) == 0:
        print("No device found")
        exit()
    
    """[
        '9E92E91B-D8FF-F21D-ADA7-48C5B560B0EA', #v3.11-1
        'E22711BB-405A-5A5D-763E-4B53F5898242', #v3.11-2
        'D10FDE72-912F-B491-B65A-9982B762036A', #v3.10
    ]"""

    rFlag=get_runFalg(imAddrs[0])
    # 接收数据函数
    drc=ImDataReceiver(imAddrs[0])
    # 绘图函数
    dDrawer=PyQtGraphDrawer(imAddrs[0])
    # 数据存储函数
    ds=DataStore(imAddrs[0],FdrName=f"testData/{MotionName}")
    #蓝牙通知函数注册
    drc.notiFxList.append(ds.checkIfDataNeedToSave)

    dDrawer.run()
    drc.run()
    print("start")

    dDrawer.app.exec_()
