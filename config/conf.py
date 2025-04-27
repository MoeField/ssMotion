#动作识别加速度阈值
ACC_Threshold=40
GYRO_Threshold=800

#动作持续帧
CONF_BUFFER_SIZE = 80  # 缓冲区(循环队列)大小
MotionPreSize = 35  # 动作峰值前数据帧数
UPDATE_EVERY_N_FRAMES=2 #每N帧更新一次绘图,必须小于CONF_BUFFER_SIZE