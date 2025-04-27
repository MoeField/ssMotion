import threading

class RunFlag:
    def __init__(self):
        self.flag = True
        self.lock = threading.Lock()

    def get(self)->bool:
        with self.lock:
            return self.flag

    def set(self, value:bool)->None:
        with self.lock:
            self.flag = value

_runFlag = {}
def get_runFalg(n:int|str=-1)->RunFlag:
    global _runFlag
    if _runFlag.get(f'{n}') is None:
        _runFlag[f'{n}'] = RunFlag()
    return _runFlag[f'{n}']