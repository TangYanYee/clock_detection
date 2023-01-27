import cv2 as cv
import numpy as np
import math
from easyocr import Reader
import serial_comm as ser
import shape_detection as shape
# reader = Reader(['en'],gpu = False)
state_dict = {
    'FIND_NUMBER':0, 
    'DETECT_HANDS':1, 
    'DETECT_HOLES':2,
    'FIND_BLACK_BOX':3,
    'FIND_FRAME':4
}
state = 6

from imutils.object_detection import non_max_suppression
vid_cam = cv.VideoCapture(0, cv.CAP_DSHOW)
vid_cam.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
vid_cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
center_list = []
small_center = (0,0)
dots_center = []
is_12_on_top = None
# this affects the size of the image the value should calculate by the actual length divide by the pixel difference in x/y
# unit in mm
#  - - > x
# |
# | screen
# v
# y

x_coff = 300/882 
y_coff = 300/882



#click fuction for printing out the pixel and calculated mm
def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        print(x*x_coff, ' ', y*y_coff)
 
 
    # checking for right mouse clicks    
    if event == cv.EVENT_RBUTTONDOWN:
        print(x, ' ', y)

if __name__=='__main__':
    num = 0
    while(vid_cam.isOpened()):
        ret, src = vid_cam.read()
        if src is not None:
            

            src, gray, thresh, binaryIMG = shape.filteringImage(src)
            src_print = src.copy()
            
            #get size of the image
            rows = src.shape[0] 
            cols = src.shape[1]
                
            if state == state_dict['FIND_NUMBER']:  
                center_list = shape.detect_circle(src = binaryIMG, src_print = src_print, minR = 340, maxR = 360)
                # Find small dots 
                if len(center_list)>=1:
                    dots_center = shape.detect_dots(thresh,src_print, center_list, 80, 150, 200, 235)
                    cv.circle(src_print, center_list[0], 235, (0, 255, 255), 2) 
                    cv.circle(src_print, center_list[0], 200, (0, 255, 255), 2) 
                    if(len(dots_center)>=2):
                        print("clock ang: ", math.atan2(dots_center[0][1]-dots_center[1][1], dots_center[0][0]-dots_center[1][0])/np.pi*180.0)
                        is_12_on_top = shape.detect_orientation(src,src_print,dots_center,center_list)
                        print(is_12_on_top)
                    # if(is_12_on_top is not None):
                    #     state += 1
 

            elif state == state_dict['DETECT_HANDS']:
                # if is_12_on_top is trueign it with the top
                # if is_12_on_top is flase, align it with the bottom
                thresh = cv.threshold(gray, 90, 255, cv.THRESH_BINARY_INV)[1] # change the image to binary image by setting a threshold 117, when change new lighting setup you should change the 2nd parameter
                #should change the below part if the position for putting the hour/min/second hand is changed
                startpointx = 1300
                startpointy = 170
                endpointx = cols-300
                endpointy = rows-300
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
                thresh = cv.threshold(gray, 7, 255, cv.THRESH_BINARY)[1]
                #find the center of the clock
                center_list = shape.detect_circle(src = binaryIMG, src_print = src_print, minR = 400, maxR = 440, votes = 30) #should adjust the min and max radius when the height of the camera changed
                if len(center_list) >= 1:
                    # ser.testing_func(1)
                    holes_center = shape.detect_dots(thresh,src_print, center_list, 100, 370, 370, 410)
                    cv.circle(src_print, center_list[0], 410, (0, 255, 255), 2) 
                    cv.circle(src_print, center_list[0], 370, (0, 255, 255), 2) 
                    if len(holes_center) == 6: # make sure all 6 holes has been detected
                        for i in range(0,6,2):
                            # calculation of the distance between 2 holes
                            distx = (holes_center[0+i][0] - holes_center[1+i][0])*x_coff 
                            disty = (holes_center[0+i][1] - holes_center[1+i][1])*y_coff
                            print("line length:",math.sqrt(distx*distx+disty*disty))
                            cv.line(src_print, holes_center[0+i], holes_center[i+1], (255,100,255), 1)
                # else:
                #     ser.testing_func(0)

            elif state == state_dict['FIND_BLACK_BOX']:  
                thresh = cv.threshold(gray, 35, 255, cv.THRESH_BINARY_INV)[1] # change the image to binary image by setting a threshold 117, when change new lighting setup you should change the 2nd parameter
                cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
                cv.drawContours(src_print,cnts[0],-1,(255,0,0),1)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts: 
                    area = cv.contourArea(c) 
                    if(area > 20000 and area < 140000):# the hour and min hands
                        print(area)
                        #use a minimum area rectangle to bound the hour and min hand
                        rect = cv.minAreaRect(c)
                        box = cv.boxPoints(rect)
                        box = np.int0(box)
                        # Retrieve the key parameters of the rotated bounding box
                        top_point_y = np.min(box[:, 1])
                        bottom_point_y = np.max(box[:, 1])
                        top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
                        bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
                        width = int(rect[1][0])
                        height = int(rect[1][1])
                        center = (int(rect[0][0]),int(rect[0][1]))
                        center_list = shape.detect_circle(src = binaryIMG, src_print = src_print, minR = 8, maxR = 15,votes = 5,circle_xy = center,mindis = 15,maxdis = 45)
                        if(area > 120000):
                            cv.drawContours(src_print,[box],0,(0,0,170),2)
                        elif(area < 40000):
                            cv.drawContours(src_print,[box],0,(0,255,255),2)
                        else:
                            cv.drawContours(src_print,[box],0,(255,255,100),2)
                        x, y, w, h = cv.boundingRect(c)

            # elif state == state_dict['FIND_FRAME']:
            #     thresh = cv.threshold(gray, 140, 255, cv.THRESH_BINARY_INV)[1] # change the image to binary image by setting a threshold 117, when change new lighting setup you should change the 2nd parameter
                

        cv.imshow('frame', binaryIMG)
        cv.imshow('thresh', thresh) 
        cv.imshow('src', src_print)
        cv.setMouseCallback('src', click_event)
        key = cv.waitKey(4)
        if (key & 0xFF == ord('q')) or (key & 0xFF == ord('Q')):
            break
        if (key & 0xFF == ord('s')) or (key & 0xFF == ord('S')):
            name_str = "hand" + str(num) +".jpg"
            cv.imwrite(name_str, src)
            num += 1
            
    vid_cam.release()
    cv.destroyAllWindows()
    print('Hi')
