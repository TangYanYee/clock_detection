import serial  # 引用pySerial模組
import struct
import serial.tools.list_ports as port_list
import time
ON = 1
OFF = 0
CLOCKWISE = 1
ANTI_CLOCKWISE = 0
COM_PORT1 = 'COM1'
COM_PORT2 = 'COM1'
ports = list(port_list.comports())
for p in ports:
    print("found: ",p,p.description)

    if(p.description.find('CH340') != -1):
        print("found2: ",p.name)
        COM_PORT2 = p.name    # 指定通訊埠名稱
    if(p.description.find('Serial Device') != -1):
        print("foun3: ",p.name)
        COM_PORT1 = p.name    # 指定通訊埠名稱
        

xyz_table = serial.Serial(
    port=COM_PORT1, baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
)
arduino = serial.Serial(
    port=COM_PORT2, baudrate=9600, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE
)

data_raw = xyz_table.read()  # 讀取一行
data_raw = arduino.read()  # 讀取一行

def send_Gcode(g_code):
    ret_msg = []
    if(isinstance(g_code, str)):
        g_code = g_code.split(",")
        print(g_code)
    for g in g_code:
        print(g)
        g = g.strip() + '\n'
        xyz_table.write(str.encode(g))
        grbl_out = xyz_table.readline() # Wait for grbl response with carriage return
        ret_msg.append(grbl_out)
        print (' : ', grbl_out.strip())
    return ret_msg
# https://item.taobao.com/item.htm?spm=a21wu.12321156-tw.go-detail.1.1509d7b0OEdcyr&id=647836280187&skuId=4671835692377
def packXYZ(arr):
    return " X"+str(arr[0])+" Y"+ str(-arr[1]) + " Z"+str(arr[2])
def packXYZ_offset(arr):
    return " X"+str(arr[0]/10.0+7.0)+" Y"+ str(-(arr[1]/10.0+7.0)) + " Z"+str(arr[2])
def move(arr):
    send_Gcode("G0" + packXYZ_offset(arr))
def move_pid(arr):
    send_Gcode("G0" + packXYZ(arr))
def drilling_action(val,val2):
    try:
        string = str(val).encode()+str(val2).encode()
        arduino.write(string)
        
        arduino = arduino.readline() # Wait for grbl response with carriage return
        arduino2 = arduino.readline() # Wait for grbl response with carriage return
        
        print ('recv: ', arduino.strip(),arduino2.strip())
        print(string)
    except:
        print("Arduino disconnected")
def initialization(): 
    time.sleep(2)   # Wait for grbl to initialize 
    xyz_table.flushInput()  # Flush startup text in serial input
    send_Gcode("G92 X0 Y0 Z0")
    send_Gcode("G90")
    send_Gcode("G0" + packXYZ([0,4,0]))
    # send_Gcode("G0" + packXYZ([6.35,5,0]))
    # send_Gcode("G0" + packXYZ([6.35,2.2,0]))
    # send_Gcode("G0" + packXYZ([6.35,2.2,-3]))
    # send_Gcode("G0" + packXYZ([6.35,4.5,-3]))
    # send_Gcode("G0" + packXYZ([6.35,4.5,0]))
initialization()
if __name__ == "__main__":
    counts = 0
    while True:    
        import keyboard
        try:
            x = float(input('x:'))
            y = float(input('y:'))
            z = float(input('z:'))
            send_Gcode("G0" + packXYZ([x,y,z]))

            # str_g = input('Gcode')
            # send_Gcode(str_g)
           
            # if keyboard.read_key() == "a":


            #     send_Gcode("G0" + packXYZ([6.3,0,0]))
            #     # counts += 1
            #     # send_Gcode("G0" + packXYZ([0,0,counts]))
            #     # drilling_action(0,0)
            #     # send_Gcode("M7")
            # if keyboard.read_key() == "q":
            #     # send_Gcode("G0" + packXYZ([0,0,counts]))
            #     if(counts == 0):
            #         send_Gcode("G0" + packXYZ([0,2,0]))
            #         send_Gcode("G0" + packXYZ([6.3,2,0]))
            #         send_Gcode("G0" + packXYZ([6.3,0,0]))
            #     counts = 1
            #     send_Gcode("G0" + packXYZ([6.3,0,-3.5]))
            # if keyboard.read_key() == "1":
            #     # counts -= 0.1
            #     send_Gcode("G0" + packXYZ([6.3,2,-3.5]))
            #     # drilling_action(1,0)
            #     # send_Gcode("M9")
            # if keyboard.read_key() == "w":
            #     send_Gcode("G0" + packXYZ([2,0,0]))
            # if keyboard.read_key() == "e":
            #     drilling_action(1,1)
            # if keyboard.read_key() == "h":
            #     send_Gcode("G0" + packXYZ([0,0,0]))
            # if keyboard.read_key() == "g":
            #     send_Gcode("G0" + packXYZ([6.3,2,0]))
      
      
            #     while True:
            #         # xyz_table.open()
            #         # try:
            #         # except:
            #         #     print('1')
            #         if(counts < 300):
            #             xyz_table.write(b'1')
            #             print('on')
            #         elif (counts < 600):
            #             xyz_table.write(b'0')
            #             print('off')
            #         else:
            #             counts = 0
            #         counts += 1
        except KeyboardInterrupt:
            xyz_table.close()    # 清除序列通訊物件
            print('再見！')
            break
