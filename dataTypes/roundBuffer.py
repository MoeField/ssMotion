import numpy as np
from threading import Lock
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
# CONFIG
from config.conf import *
floatType = np.float32
# END CONFIG

# 定义数据类型
dataType_XYZ = np.dtype([
    ('X', floatType),
    ('Y', floatType),
    ('Z', floatType)
])

dataType_QUAT = np.dtype([
    ('W', floatType),
    ('X', floatType),
    ('Y', floatType),
    ('Z', floatType)
])

dataType_ImuFrame = np.dtype([
    ('Dtag', np.uint16),
    ('timeStamp', np.uint32),
    ('acc', dataType_XYZ),
    ('ACC', dataType_XYZ),
    ('gyro', dataType_XYZ),
    ('mag', dataType_XYZ),
    ('temp', floatType),
    ('press', floatType),
    ('height', floatType),
    ('quat', dataType_QUAT),
    ('angle', dataType_XYZ),
    ('offset', dataType_XYZ),
    ('doWalk', np.bool_),
    ('doRun', np.bool_),
    ('doBike', np.bool_),
    ('doCar', np.bool_),
    ('acs', dataType_XYZ),
    ('voltage', floatType),
    ('gpio', np.uint8),
], align=True)


# 定义环形缓冲区类
class RoundBuffer:
    def __init__(self, 
        dataType = dataType_ImuFrame,
        size=CONF_BUFFER_SIZE
    ):
        self.lock = Lock()
        self.size = size
        self.dataType = dataType
        self.ptr = 0 # 指向当前位置
        self.data = np.zeros(size, dtype=dataType)

    def push(self, data:dataType_ImuFrame)->None:
        with self.lock:
            self.data[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.size

    def getAll(self)->np.ndarray:
        with self.lock:
            if self.ptr == 0:
                return self.data.copy()
            else:
                return np.concatenate(
                    (self.data[self.ptr:], self.data[:self.ptr])
                )
    
    def clear(self)->None:
        with self.lock:
            self.data = np.zeros(self.size, dtype=self.dataType)
            self.ptr = 0
        
    
# global shared buffer
_dataBuffer:dict = {}

def get_dataBuffer(n:int|str=-1)->RoundBuffer:
    global _dataBuffer
    if _dataBuffer.get(f'{n}') is None:
        _dataBuffer[f'{n}'] = RoundBuffer()
    return _dataBuffer[f'{n}']

#test
if __name__ == '__main__':
    tframeData = np.array((
        1,14,
        (1,2,3),
        (4,5,6),
        (7,8,9),
        (10,11,12),
        13,14,15,
        (21,22,23,24),
        (25,26,27),
        (28,29,30),
        True,False,True,False,
        (31,32,33),
        34,35
    ), dtype=dataType_ImuFrame)

    buffer = RoundBuffer()
    buffer.push(tframeData)
    buffer.push(tframeData)
    print(buffer.getAll())
    for i in range(10):
        buffer.push(tframeData)
    print(buffer.getAll())
    buffer.clear()
    print(buffer.getAll())
    for i in range(79):
        buffer.push(tframeData)
    print(buffer.getAll())
    buffer.clear()
    print(buffer.getAll())