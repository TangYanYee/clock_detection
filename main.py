import cv2 as cv
import numpy as np
import math
from easyocr import Reader
import serial_comm as xyz
import shape_detection as shape
from vidgear.gears import VideoGear
from vidgear.gears import CamGear
import time

options1 = {
    "CAP_PROP_FRAME_WIDTH": 2560, # resolution 320x240
    "CAP_PROP_FRAME_HEIGHT": 1440
}
options2 = {
    "CAP_PROP_FRAME_WIDTH": 960, # resolution 320x240
    "CAP_PROP_FRAME_HEIGHT": 720
}

# To open live video stream on webcam at first index(i.e. 0) 
# device and apply source tweak parameters
stream1 = CamGear(source=0, logging=True, **options1).start()
stream2 = CamGear(source=1, logging=True, **options2).start()

state_dict = {
    'FIND_NUMBER':0, 
    'DETECT_HANDS':1, 
    'DETECT_HOLES':2,
    'FIND_BLACK_BOX':3,
    'PRINT':10,
}



state = 3
center_list = []
small_center = (0,0)
dots_center = []
# this affects the size of the image the value should calculate by the actual length divide by the pixel difference in x/y
# unit in mm
#  - - > x
# | screen
# |
# v
# y

# xyz table coordinate system
# y
# ^
# | screen
# |
#  - - > x
def align_hole(hole_idx):
    do_once = 0
    while(1):
        src2 = stream2.read( )# capture the webcam
        if src2 is not None:   
            #for debugging 
            if(do_once == 0):
                name_str = 'hole' + str(hole_idx) + ".jpg"
                cv.imwrite(name_str, src2) # export the webcam as an image
                do_once = 1
            src2, gray2, thresh2, binaryIMG2 = shape.filteringImage2(src2)
            dots_center = shape.detect_dots(thresh2,src2, [(0,0)], 2000, 10000, 0, 700000)
            cv.imshow('thresh2', thresh2) #display the photo
            cv.imshow('src2', src2) #display the photo
            cv.waitKey(4)
            print("Cam2: ",dots_center)
            # # -12.5
            #aligning the center using pid control, with only p term
            if(len(dots_center) == 1):
                xyz.send_Gcode("G91")
                x = (dots_center[0][0] - 145) * 1/1000 * 0.9
                y = -(dots_center[0][1] - 139) * 1/1000 * 0.9
                if(abs(dots_center[0][0] - 145) < 3):
                    x = 0
                if(abs(dots_center[0][1] - 139) < 3):
                    y = 0
                xyz.move_pid([x,y,0])
                time.sleep(0.1)
                if x == 0 and y == 0:
                    print('finish_align')
                    break            
#click fuction for printing out the pixel and calculated mm
if __name__=='__main__':
    num = 0
    while(1):
        src = stream1.read()

            # 289 250
             
        if src is not None:
            

            src, gray, thresh, binaryIMG = shape.filteringImage(src)
            src_print = src.copy()
            
            #get size of the image
            rows = src.shape[0] 
            cols = src.shape[1]
            if state == state_dict['FIND_NUMBER']:  
                thresh = cv.threshold(gray, 130, 255, cv.THRESH_BINARY_INV)[1] # change the image to binary image by setting a threshold 117, when change new lighting setup you should change the 2nd parameter
                center_list = shape.detect_circle(src = binaryIMG, src_print = src_print, minR = 245, maxR = 265)
                # Find small dots 
                if len(center_list)>=1:
                    dots_center = shape.detect_dots(thresh,src_print, center_list, 5, 40, 145, 175)
                    cv.circle(src_print, center_list[0], 175, (0, 255, 255), 2) 
                    cv.circle(src_print, center_list[0], 145, (0, 255, 255), 2) 

                    if(len(dots_center)>= 12):
                        point = shape.detect_orientation(src,src_print,dots_center,center_list)
                        print("clock ang: ", math.atan2(point[0][1]-point[1][1], point[0][0]-point[1][0])/np.pi*180.0)
                        print(point)
 

            elif state == state_dict['DETECT_HANDS']:
                thresh = cv.threshold(gray, 190, 255, cv.THRESH_BINARY_INV)[1] # change the image to binary image by setting a threshold 117, when change new lighting setup you should change the 2nd parameter
                #should change the below part if the position for putting the hour/min/second hand is changed
                startpointx = 1300
                startpointy = 0
                endpointx = cols-750
                endpointy = rows-1200
                # crop the image
                img = thresh[startpointy:endpointy,startpointx:endpointx] #cut out part of the threshold image
                cv.rectangle(src_print,(startpointx,startpointy),(endpointx,endpointy),(0,0,0),1)

                # To sort the list in place...
                hand_arr = shape.detect_hand(img, src_print)
                hand_arr.sort(key=lambda x: x.length, reverse=False)
                if(len(hand_arr) == 3):
                    print("Hour Hand:")
                    hand_arr[0].show_data()
                    print("Min Hand:")
                    hand_arr[1].show_data()
                    print("Sec Hand:")
                    hand_arr[2].show_data()

            elif state == state_dict['DETECT_HOLES']:   
                thresh = cv.threshold(gray, 40, 255, cv.THRESH_BINARY)[1]
                #find the center of the clock
                center_list = shape.detect_circle(src = binaryIMG, src_print = src_print, minR = 300, maxR = 320, votes = 25) #should adjust the min and max radius when the height of the camera changed
                if len(center_list) >= 1:
                    # xyz_table.drilling_action(1)
                    holes_center = shape.detect_dots(thresh,src_print, center_list, 80, 300, 265, 300)
                    cv.circle(src_print, center_list[0], 300, (0, 255, 255), 2) 
                    cv.circle(src_print, center_list[0], 265, (0, 255, 255), 2) 
                    print(shape.pixel2xy([center_list[0][0],center_list[0][1]]))
                    if len(holes_center) == 6: # make sure all 6 holes has been detected
                        name_str = "move"  +".jpg"
                        cv.imwrite(name_str, src_print)
                        for i in range(0,6,2):
                            # calculation of the distance between 2 holes
                            distx = (holes_center[0+i][0] - holes_center[1+i][0])*shape.x_coff 
                            disty = (holes_center[0+i][1] - holes_center[1+i][1])*shape.x_coff
                            print("line length:",math.sqrt(distx*distx+disty*disty))
                            if(math.sqrt(distx*distx+disty*disty)<150):    
                                print("line coor: ", holes_center[0+i], ",",holes_center[i+1])  
                                cv.line(src_print, holes_center[0+i], holes_center[i+1], (255,100,255), 1)
                        hole_idx = 0
                        for i in holes_center:
                            xyz.send_Gcode("G90")
                            xyz.move(shape.xy2xyz(shape.pixel2xy([i[0],i[1]]),0))
                            # xyz.move(shape.xy2xyz(shape.pixel2xy([i[0],i[1]]),-5))
                            time.sleep(10)
                            align_hole(hole_idx)
                            hole_idx += 1
                            # xyz.move_pid([0,0,-12.5])
                            xyz.move_pid([0,0,-5])
                            xyz.drilling_action(xyz.ON,xyz.CLOCKWISE)
                            time.sleep(2)
                            # xyz.move_pid([0,0,12.5])
                            xyz.move_pid([0,0,5])
                            xyz.drilling_action(xyz.OFF,xyz.CLOCKWISE)

                        state = 10

                # else:
                #     xyz_table.drilling_action(0)

            elif state == state_dict['FIND_BLACK_BOX']:  
                outer_box ,clock_movement = shape.detect_black_box(src,src_print)
                thresh = cv.threshold(gray, 90, 255, cv.THRESH_BINARY_INV)[1] # change the image to binary image by setting a threshold 117, when change new lighting setup you should change the 2nd parameter
                if(len(outer_box)>0):
                    cv.rectangle(src_print,(outer_box[0],outer_box[1]), (outer_box[2],outer_box[3]), (0, 255, 0), 4)
                    img = thresh[outer_box[1]:outer_box[3],outer_box[0]:outer_box[2]] #cut out part of the threshold image
                if(len(clock_movement)>0):
                    cv.rectangle(src_print,(clock_movement[0],clock_movement[1]), (clock_movement[2],clock_movement[3]), (255, 0, 0), 4)
                    img2 = thresh[clock_movement[1]:clock_movement[3],clock_movement[0]:clock_movement[2]] #cut out part of the threshold image

        src_print = cv.resize(src_print, (1920,1080), interpolation = cv.INTER_AREA)
        cv.imshow('frame', binaryIMG)
        cv.imshow('thresh', thresh) 
        cv.imshow('src', src_print)
        cv.setMouseCallback('src', shape.click_event)
        key = cv.waitKey(4)
        if (key & 0xFF == ord('q')) or (key & 0xFF == ord('Q')):
            break
        if (key & 0xFF == ord('s')) or (key & 0xFF == ord('S')):
            name_str = "clock1" + str(num) +".jpg"
            cv.imwrite(name_str, src)
            num += 1
    
    # vid_cam.release()
    stream1.stop()
    stream2.stop()
    cv.destroyAllWindows()
    print('Hi')
