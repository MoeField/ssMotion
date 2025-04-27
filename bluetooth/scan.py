import asyncio
from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic

#设备的MAC地址
par_device_addrs = []

async def fetchMacAddr(
        dName:str = "im948-V3.1",        # 设备名称开头部分
        toWhere:list = par_device_addrs   # 存储结果的列表
    ):
    flagFound=False

    print("scan Devices...")
    #扫描设备,存储结果
    devices = await BleakScanner.discover()

    print(f"Found {len(devices)} devices:")
    for d in devices:
        print(d)
        if d.name is None:  continue
        if dName in d.name:
            flagFound=True
            toWhere.append(d.address)
    del devices

    if not flagFound:
        #print(f"could not find device with name {dName}")
        return None
    
    return toWhere

if __name__ == "__main__":
    print("start fetchMacAddr...")
    imAddrs = []
    asyncio.run(fetchMacAddr(toWhere=imAddrs))
    if len(imAddrs) > 0:
        print(f"found {len(imAddrs)} devices with name im948-V3.1")