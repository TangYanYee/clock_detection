import serial  # 引用pySerial模組
import struct
import sys
import time
import serial.tools.list_ports as port_list

COM_PORT = 'COM1'
ports = list(port_list.comports())
for p in ports:
    if(p.description.find('CH340')):
        print("found: ",p.name)
        COM_PORT = p.name    # 指定通訊埠名稱
ser = serial.Serial(
    port=COM_PORT, baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
)
data_raw = ser.read()  # 讀取一行
def send_xyz_to_pc(x, y, z):
    # ser.write()
    xyz_data = struct.pack('>HHH',x,y,z)
    ser.write(xyz_data)
    print(xyz_data)

def testing_func(val):
    if(val):
        ser.write(b'1')
        print('on')
    else:
        ser.write(b'0')
        print('off')
if __name__ == "__main__":
    counts = 0
    try:
      
        while True:
            # ser.open()
            # try:
            # except:
            #     print('1')
            if(counts < 300):
                ser.write(b'1')
                print('on')
            elif (counts < 600):
                ser.write(b'0')
                print('off')
            else:
                counts = 0
            counts += 1
    except KeyboardInterrupt:
        ser.close()    # 清除序列通訊物件
        print('再見！')