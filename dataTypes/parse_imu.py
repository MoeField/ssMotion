import numpy as np
from array import array

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from dataTypes.roundBuffer import dataType_ImuFrame # 导入自定义的数据类型

# 定义缩放因子
scaleAccel       = 0.00478515625      # 加速度 [-16g~+16g]    9.8*16/32768
scaleQuat        = 0.000030517578125  # 四元数 [-1~+1]         1/32768
scaleAngle       = 0.0054931640625    # 角度   [-180~+180]     180/32768
scaleAngleSpeed  = 0.06103515625      # 角速度 [-2000~+2000]    2000/32768
scaleMag         = 0.15106201171875   # 磁场 [-4950~+4950]   4950/32768
scaleTemperature = 0.01               # 温度
scaleAirPressure = 0.0002384185791    # 气压 [-2000~+2000]    2000/8388608
scaleHeight      = 0.0010728836       # 高度 [-9000~+9000]    9000/8388608
############################################################################

# imu数据解析
def parse_imu(buf:bytearray):
    if len(buf) < 7:# 最小预期数据长度，如果小于这个长度，说明数据一定不完整
        #print(f"[@parse_imu] invalid data length:{len(buf)}, expected:>7")
        return None

    #imu_dat = array('f',[0.0 for i in range(0,34)])
    imu_dat = np.zeros(34, dtype=np.float32)
    tag=None
    timeStamp=None

    # IMU数据解析
    if buf[0] == 0x11:
        ctl = (buf[2] << 8) | buf[1]
        #tag= "0x%04x"%ctl
        timeStamp=((buf[6]<<24) | (buf[5]<<16) | (buf[4]<<8) | (buf[3]<<0))
        #timeStamp=((buf[7]<<24) | (buf[8]<<16) | (buf[9]<<8) | (buf[10]<<0))#ai
        #print("ms:",timeStamp)

        L =7 # 从第7字节开始根据 订阅标识tag来解析剩下的数据
        if ((ctl & 0x0001) != 0):
            tmpX = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAccel; L += 2 
            tmpY = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAccel; L += 2 
            tmpZ = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAccel; L += 2

            imu_dat[0] = float(tmpX)# x加速度aX
            imu_dat[1] = float(tmpY)# y加速度aY
            imu_dat[2] = float(tmpZ)# z加速度aZ
        
        if ((ctl & 0x0002) != 0):
            tmpX = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAccel; L += 2
            tmpY = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAccel; L += 2
            tmpZ = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAccel; L += 2

            imu_dat[3] = float(tmpX)# x加速度AX
            imu_dat[4] = float(tmpY)# y加速度AY
            imu_dat[5] = float(tmpZ)# z加速度AZ

        if ((ctl & 0x0004) != 0):
            tmpX = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAngleSpeed; L += 2 
            tmpY = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAngleSpeed; L += 2 
            tmpZ = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAngleSpeed; L += 2

            imu_dat[6] = float(tmpX)# x角速度GX
            imu_dat[7] = float(tmpY)# y角速度GY
            imu_dat[8] = float(tmpZ)# z角速度GZ
        
        if ((ctl & 0x0008) != 0):
            tmpX = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleMag; L += 2
            tmpY = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleMag; L += 2
            tmpZ = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleMag; L += 2

            imu_dat[9] = float(tmpX)# x磁场CX
            imu_dat[10] = float(tmpY)# y磁场CY
            imu_dat[11] = float(tmpZ)# z磁场CZ
        
        if ((ctl & 0x0010) != 0):
            tmpX = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleTemperature; L += 2
            tmpU32 = np.uint32(((np.uint32(buf[L+2]) << 16) | (np.uint32(buf[L+1]) << 8) | np.uint32(buf[L])))
            if ((tmpU32 & 0x800000) == 0x800000): # 若24位数的最高位为1则该数值为负数，需转为32位负数，直接补上ff即可
                tmpU32 = (tmpU32 | 0xff000000)      
            tmpY = np.int32(tmpU32) * scaleAirPressure; L += 3
            tmpU32 = np.uint32((np.uint32(buf[L+2]) << 16) | (np.uint32(buf[L+1]) << 8) | np.uint32(buf[L]))
            if ((tmpU32 & 0x800000) == 0x800000): # 若24位数的最高位为1则该数值为负数，需转为32位负数，直接补上ff即可
                tmpU32 = (tmpU32 | 0xff000000)
            tmpZ = np.int32(tmpU32) * scaleHeight; L += 3 

            imu_dat[12] = float(tmpX)# 温度
            imu_dat[13] = float(tmpY)# 气压
            imu_dat[14] = float(tmpZ)# 高度

        if ((ctl & 0x0020) != 0):
            tmpAbs = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleQuat; L += 2
            tmpX =   np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleQuat; L += 2
            tmpY =   np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleQuat; L += 2
            tmpZ =   np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleQuat; L += 2

            imu_dat[15] = float(tmpAbs) # w
            imu_dat[16] = float(tmpX)# x
            imu_dat[17] = float(tmpY)# y
            imu_dat[18] = float(tmpZ)# z

        if ((ctl & 0x0040) != 0):
            tmpX = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAngle; L += 2
            tmpY = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAngle; L += 2
            tmpZ = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAngle; L += 2

            imu_dat[19] = float(tmpX) # x角度
            imu_dat[20] = float(tmpY) # y角度
            imu_dat[21] = float(tmpZ) # z角度

        if ((ctl & 0x0080) != 0):
            tmpX = np.short((np.short(buf[L+1])<<8) | buf[L]) / 1000.0; L += 2
            tmpY = np.short((np.short(buf[L+1])<<8) | buf[L]) / 1000.0; L += 2
            tmpZ = np.short((np.short(buf[L+1])<<8) | buf[L]) / 1000.0; L += 2

            imu_dat[22] = float(tmpX) # x坐标
            imu_dat[23] = float(tmpY) # y坐标
            imu_dat[24] = float(tmpZ) # z坐标

        if ((ctl & 0x0100) != 0):
            tmpU32 = ((buf[L+3]<<24) | (buf[L+2]<<16) | (buf[L+1]<<8) | (buf[L]<<0)); L += 4
            #iPrint("\tsteps: %u"%tmpU32); # 计步数
            tmpU8 = buf[L]; L += 1
            imu_dat[25] = 100 if (tmpU8 & 0x01) else 0 # 是否在走路
            imu_dat[26] = 100 if (tmpU8 & 0x02) else 0 # 是否在跑步
            imu_dat[27] = 100 if (tmpU8 & 0x04) else 0 # 是否在骑车
            imu_dat[28] = 100 if (tmpU8 & 0x08) else 0 # 是否在开车

        if ((ctl & 0x0200) != 0):
            tmpX = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAccel; L += 2
            tmpY = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAccel; L += 2
            tmpZ = np.short((np.short(buf[L+1])<<8) | buf[L]) * scaleAccel; L += 2
        
            imu_dat[29] = float(tmpX)# x加速度asX
            imu_dat[30] = float(tmpY)# y加速度asY
            imu_dat[31] = float(tmpZ)# z加速度asZ
            
        if ((ctl & 0x0400) != 0):
            tmpU16 = ((buf[L+1]<<8) | (buf[L]<<0)); L += 2
            imu_dat[32] = float(tmpU16)# adc测量到的电压值，单位为mv

        if ((ctl & 0x0800) != 0):
            tmpU8 = buf[L]; L += 1
            #iPrint("\t GPIO1  M:%X, N:%X"%((tmpU8>>4)&0x0f, (tmpU8)&0x0f))
            imu_dat[33] = float(tmpU8)

        try:
            imu_dict_data = np.array((
                ctl,timeStamp,          # 标签 时间戳
                (imu_dat[0],imu_dat[1],imu_dat[2]),     # 加速度
                (imu_dat[3],imu_dat[4],imu_dat[5]),     # 加速度
                (imu_dat[6],imu_dat[7],imu_dat[8]),     # 角速度
                (imu_dat[9],imu_dat[10],imu_dat[11]),   # 磁场
                imu_dat[12],imu_dat[13],imu_dat[14],    # 温度 气压 高度
                (imu_dat[15],imu_dat[16],imu_dat[17],imu_dat[18]),  # 四元数
                (imu_dat[19],imu_dat[20],imu_dat[21]),  # 角度
                (imu_dat[22],imu_dat[23],imu_dat[24]),  # 坐标(位置偏移)
                True if imu_dat[25] else False,
                True if imu_dat[26] else False,
                True if imu_dat[27] else False,
                True if imu_dat[28] else False,
                (imu_dat[29],imu_dat[30],imu_dat[31]),  # 加速度
                imu_dat[32],imu_dat[33],                # 电压值  GPIO1
            ), dtype=dataType_ImuFrame)

            return imu_dict_data
        except Exception as e:
            print(f"[@parse_imu] error:{e}")
            exit(0)
            return None
        
    else:
        print("[error] data head not define")
        return None

if __name__ == "__main__":
    print("this is a module, not a script")