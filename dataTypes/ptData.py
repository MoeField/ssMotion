import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from threading import Lock

from src.motionPredictor import SVMActionPredictor, CnnLstmActionPredictor

class PtData:
    def __init__(self, imAddr:str='test'):
        self.lock = Lock()
        self.imAddr = imAddr
        self.data = pd.DataFrame()
        self.svmResultStr = ""
        self.svmResultConfident = 0.0
        self.cnnLstmResultStr = ""
        self.cnnLstmResultConfident = 0.0

        self.svmP = SVMActionPredictor()
        self.cnnLstmP = CnnLstmActionPredictor()

    def update(self, data:pd.DataFrame):
        with self.lock:
            self.data = data
            self.svmResultStr, self.svmResultConfident = self.svmP.predict(data)
            self.cnnLstmResultStr, self.cnnLstmResultConfident = self.cnnLstmP.predict(data)
        
    def getData(self):
        with self.lock:
            return self.data.copy()
    
    def getSvmResult(self)->tuple[str, float]:
        with self.lock:
            return self.svmResultStr, self.svmResultConfident
    
    def getCnnLstmResult(self)->tuple[str, float]:
        with self.lock:
            return self.cnnLstmResultStr, self.cnnLstmResultConfident

ptDataDict = {}
def get_ptData(imAddr:str='test')->PtData:
    if imAddr not in ptDataDict:
        ptDataDict[imAddr] = PtData(imAddr)
    return ptDataDict[imAddr]
    

