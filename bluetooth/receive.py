import asyncio
import threading
import time
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic
import numpy as np

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from bluetooth.scan import fetchMacAddr
from dataTypes.runFlag import get_runFalg
from dataTypes.roundBuffer import dataType_ImuFrame, RoundBuffer, get_dataBuffer
from dataTypes.parse_imu import parse_imu

from config.conf import *
RCV_FPS = 60 # 上报频率


##############################################################
par_notification_characteristic=0x0007
par_write_characteristic=0x0005
##############################################################
# 参数设置
isCompassOn = 0 #使用磁场融合姿态
barometerFilter = 2
Cmd_ReportTag = 0x0FFF # 功能订阅标识
params = bytearray([0x00 for _ in range(0,11)])
params[0] = 0x12
params[1] = 5       #静止状态加速度阀值
params[2] = 255     #静止归零速度(单位cm/s) 0:不归零 255:立即归零
params[3] = 0       #动态归零速度(单位cm/s) 0:不归零
params[4] = ((barometerFilter&3)<<1) | (isCompassOn&1);   
params[5] = RCV_FPS #数据主动上报的传输帧率[取值0-250HZ], 0表示0.5HZ
params[6] = 1       #陀螺仪滤波系数[取值0-2],数值越大越平稳但实时性越差
params[7] = 3       #加速计滤波系数[取值0-4],数值越大越平稳但实时性越差
params[8] = 5       #磁力计滤波系数[取值0-9],数值越大越平稳但实时性越差
params[9] = Cmd_ReportTag&0xff
params[10] = (Cmd_ReportTag>>8)&0xff
##############################################################

class ImDataReceiver:
    def __init__(self, imAddr:str):
        self.notiFxList = []
        self.addr = imAddr
        self.rFlg = get_runFalg(imAddr)
        self.dataBuffer:RoundBuffer = get_dataBuffer(imAddr)
        self.updateEveryNFrames = UPDATE_EVERY_N_FRAMES
        self.framePassed = 0
        #self.lastUpdateTime = time.time()
    
    def run(self):
        #asyncio.run(self.receiveData())
        threading.Thread(target=asyncio.run,args=(self.receiveData(),)).start()
    
    def stop(self):
        self.rFlg.set(False)

##############################################################
#                       通知回调函数
##############################################################
    def notification_handler(self, 
        characteristic: BleakGATTCharacteristic, data: bytearray
    ):
        imuData = parse_imu(data)
        if imuData is None:
            print("parse_imu failed")
            return
        self.dataBuffer.push(imuData)
        self.framePassed += 1
        if self.framePassed % UPDATE_EVERY_N_FRAMES == 0:
            self.framePassed = 0

            for fx in self.notiFxList:
                fx()
##############################################################

    def disconnected_callback(self,client):
        print("Disconnected callback called!")
        self.disconnected_event.set()
        self.rFlg.set(False)
        time.sleep(1)
        return -1

    async def receiveData(self):
        framePassed = 0

        print(f"connecting to device with address {self.addr}")
        device = await BleakScanner.find_device_by_address(
            self.addr, cb={'use_bdaddr': False}  #基于MAC地址查找设备
        )
        if device is None:
            print(f"could not find device with address {self.addr}")
            return -2
        #事件定义
        self.disconnected_event = asyncio.Event()
        #断开连接事件回调

        print("connecting to device...")
        async with BleakClient(
            device,disconnected_callback=self.disconnected_callback
        ) as client:
            print("Connected")
            # 保持连接 0x29
            wakestr=bytes([0x29])
            await client.write_gatt_char(par_write_characteristic, wakestr)
            await asyncio.sleep(0.2)

            # 尝试采用蓝牙高速通信特性 0x46
            fast=bytes([0x46])
            await client.write_gatt_char(par_write_characteristic, fast)
            await asyncio.sleep(0.2)

            # GPIO 上拉
            #upstr=bytes([0x27,0x10])
            #await client.write_gatt_char(par_write_characteristic, upstr)
            #await asyncio.sleep(0.2)

            await client.write_gatt_char(par_write_characteristic, params)
            await asyncio.sleep(0.2)

            notes=bytes([0x19])
            await client.write_gatt_char(par_write_characteristic, notes)
            
            #await asyncio.sleep(0.2)
            #await client.write_gatt_char(par_write_characteristic, bytes([0x51,0xAA,0xBB])) # 用总圈数代替欧拉角传输 并清零圈数 0x51
            #await client.write_gatt_char(par_write_characteristic, bytes([0x51,0x00,0x00])) # 输出欧拉角 0x51
            print("------------------------------------------------")
            await client.start_notify(
                par_notification_characteristic, self.notification_handler
            )#注册通知回调函数


            # 添加一个循环，使程序在接收数据时不会退出
            while not self.disconnected_event.is_set():
                if self.rFlg.get() == False:
                    print("receiveData exit")
                    return
                await asyncio.sleep(0.5)
                
            await self.disconnected_event.wait() #休眠直到设备断开连接，有延迟。此处为监听设备直到断开为止
            await client.stop_notify(par_notification_characteristic)

# 测试
if __name__ == "__main__":
    # 运行扫描函数
    d=[]
    asyncio.run(fetchMacAddr(toWhere=d))
    if len(d) == 0:
        print("No device found")
        exit()
    # 运行接收数据函数
    
    drc=ImDataReceiver(d[0])
    def printData():
        print(drc.dataBuffer.getAll())
    
    drc.notiFxList.append(printData)
    drc.run()
    time.sleep(10)
    drc.stop()
    time.sleep(1)
    print("end")