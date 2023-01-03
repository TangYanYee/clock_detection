import sys
import cv2 as cv
import numpy as np
import math
import os
from easyocr import Reader
import serial_comm as ser
import shape_detection as shape
# reader = Reader(['en'],gpu = False)
state_dict = {'FIND_CIRCLE': 0, 'DETECTING_NUMBER':1, 'DETECT_HANDS':2, 'DETECT_HOLES':3,'BLACK_BOX_DETECTION':4}
state = 3
from imutils.object_detection import non_max_suppression
vid_cam = cv.VideoCapture(0, cv.CAP_DSHOW)
# vid_cam.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# vid_cam.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
center_list = []
small_center = (0,0)
dots_center = []
is_12_on_top = True
# this affects the size of the image the value should calculate by the actual length divide by the pixel difference in x/y
# unit in mm
#  - - > x
# |
# | screen
# v
# y
x_coff = 300/383 
y_coff = 300/374


#click fuction for printing out the pixel and calculated mm
def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        print(x*x_coff, ' ', y*y_coff)
 
 
    # checking for right mouse clicks    
    if event==cv.EVENT_RBUTTONDOWN:
        print(x, ' ', y)

if __name__=='__main__':
    while(vid_cam.isOpened()):
        ret, src = vid_cam.read()
        if src is not None:
            

            # ser.send_xyz_to_pc(1,2,3)
            src, gray, thresh, binaryIMG = shape.filteringImage(src)
            src_print = src.copy()
            
            #get size of the image
            rows = src.shape[0] 
            cols = src.shape[1]
            if state == state_dict['FIND_CIRCLE']:
                # Find clock circle (the white part)
                center_list = shape.detect_circle(src = binaryIMG, src_print = src_print, minR = 140, maxR = 155)
                if len(center_list) >=1 :
                    state += 1 # go to next step
                
            elif state == state_dict['DETECTING_NUMBER']:  
                # Find small dots 
                small_center = shape.detect_dots(thresh,src_print,cv.RETR_TREE,(center_list, 23, 120, 0, 20))
                if(len(small_center)>=1):
                    print("small center", small_center)
                    dots_center = shape.detect_dots(thresh,src_print,cv.RETR_TREE,(small_center, 5, 30, 85, 100))
                    cv.circle(src_print, small_center[0], 90, (0, 255, 255), 2) 
                    cv.circle(src_print, small_center[0], 100, (0, 255, 255), 2) 
                    if(len(dots_center)>=1):
                        is_12_on_top = shape.orientation_detect(src,src_print,dots_center,small_center)
                        print(is_12_on_top)
                        if(is_12_on_top is not None):
                            state += 1
 

            elif state == state_dict['DETECT_HANDS']:
                # if is_12_on_top is trueign it with the top
                # if is_12_on_top is flase, align it with the bottom
                # crop the image
                center_of_mass = 0
                print("clock ang: ", math.atan2(dots_center[0][1]-dots_center[1][1], dots_center[0][0]-dots_center[1][0])/np.pi*180.0)
                #should change the below part if the position for putting the hour/min/second hand is changed
                startpointx = 500
                startpointy = 100
                endpointx = cols
                endpointy = rows-150
                img = thresh[startpointy:endpointy,startpointx:endpointx] #cut out part of the threshold image
                img2 = binaryIMG[startpointy:endpointy,startpointx:endpointx] 
                img3 = np.zeros((endpointy-startpointy, endpointx-startpointx, 3), dtype = np.uint8)
                img3 = cv.cvtColor(img3,cv.COLOR_BGR2GRAY)
                cv.rectangle(src_print,(startpointx,startpointy),(endpointx,endpointy),(0,0,0),1)

                #find contours of 3 hands
                cnts = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
                cv.drawContours(src_print, cnts[0], -1, (255, 255, 255), 2)
                cv.drawContours(img3, cnts[0], -1, (255, 255, 255), 1)
                cv.imshow("img3", img3)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                hand_length = []
                hand_ang = []
                hand_ori = []
                for c in cnts: 
                    area = cv.contourArea(c) 
                    if(area > 500):# the hour and min hands
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
                        # because the width and height will change according to different orientation, therefore i take the longer side to be the length of the hand
                        hand_length.append(max(width,height))
                        # determine the angle by whether the width is longer or the height is longer
                        if(width > height):
                            a = rect[2] - 90
                            hand_ang.append(a)
                        else:   
                            hand_ang.append(rect[2])
                        cv.drawContours(src_print,[box],0,(255,0,0),1)
                        #find the small circles of the hand to determine the orientaion of the hand
                        circles = cv.HoughCircles(img3, cv.HOUGH_GRADIENT, 1, rows / 8,
                                                param1=255, param2=5,
                                                minRadius=4, maxRadius=7)
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for i in circles[0, :]:
                                circle_center = (i[0], i[1])
                                radius = i[2]
                                if(math.dist((top_point_x,top_point_y),circle_center)<12 or math.dist((bottom_point_x,bottom_point_y),circle_center)<12):
                                    cv.circle(src_print, circle_center, radius, (255, 0, 255), 2)
                                    if(center[1]>circle_center[1]):    
                                        hand_ori.append([1])
                                    else:
                                        hand_ori.append([0])
                    
                    elif (area > 200): # the second hand
                        m = cv.moments(c)
                        #find the center of mass to determine the orientation of the second hand, becuase the circle of the second hand is hard to detect
                        if( m["m00"] != 0):
                            x = int(m["m10"] / m["m00"])
                            y = int(m["m01"] / m["m00"])
                            center_of_mass = (x,y)
                            cv.circle(src_print, (int(x),int(y)), 1, (255, 0, 255), 1)
                if len(hand_length)>0:
                    print("hand ori:", hand_ori)
                    print("Min Hand ang: ",hand_ang[hand_length.index(max(hand_length))])
                    print("Hour Hand ang: ",hand_ang[hand_length.index(min(hand_length))])
                linesP = cv.HoughLinesP(img2, 1, np.pi / 180, 15, None, 125, 10)
                # if more then one line is detected then will calculate the mean of the 2 end point of the line, so that only one line will always appears
                if linesP is not None:
                    sum_x1 = 0
                    sum_x2 = 0
                    sum_y1 = 0
                    sum_y2 = 0
                    for i in range(0, len(linesP)):
                        l = linesP[i][0]
                        sum_x1 += l[0] 
                        sum_y1 += l[1]
                        sum_x2 += l[2]
                        sum_y2 += l[3]
                    l[0] = int(sum_x1 / len(linesP))
                    l[1] = int(sum_y1 / len(linesP))
                    l[2] = int(sum_x2 / len(linesP))
                    l[3] = int(sum_y2 / len(linesP))
                    if(center_of_mass != 0):
                        if(math.dist(center_of_mass,(l[0],l[1])) < math.dist(center_of_mass,(l[2],l[3]))):
                            far_point = (l[2],l[3])
                        else:
                            far_point = (l[0],l[1])

                        cv.circle(src_print, far_point, 1, (255, 255, 50), 3)
                    print("point1: ",l[0],",",l[1],"|| point2: ",l[2],",",l[3],"|| cg:",center_of_mass)
                    print("Sec Hand ang: ", math.atan2(l[1]-l[3], l[0]-l[2])/np.pi*180.0 +90.0)
                    cv.line(src_print, (l[0]+500, l[1]+100), (l[2]+500, l[3]+100), (0,0,255), 3, cv.LINE_AA)
                    cv.circle(src_print, (l[0], l[1]), 1, (0,0,255), 3)
                    cv.circle(src_print, (l[2], l[3]), 1, (255, 255, 50), 3)
            elif state == state_dict['DETECT_HOLES']:   
                thresh = cv.threshold(gray, 7, 255, cv.THRESH_BINARY)[1] # change the image to binary image by setting a threshold 117, when change new lighting setup you should change the 2nd parameter
                #find the center of the clock
                center_list = shape.detect_circle(src = binaryIMG, src_print = src_print, minR = 170, maxR = 185, votes = 25) #should adjust the min and max radius when the height of the camera changed
                if len(center_list) >= 1:
                    ser.testing_func(1)
                    holes_center = shape.detect_dots(thresh,src_print,cv.RETR_TREE, (center_list, 30, 100, 155, 175))
                    cv.circle(src_print, center_list[0], 155, (0, 255, 255), 2) 
                    cv.circle(src_print, center_list[0], 175, (0, 255, 255), 2) 
                    if len(holes_center) == 6: # make sure all 6 holes has been detected
                        for i in range(0,6,2):
                            # calculation of the distance between 2 holes
                            distx = (holes_center[0+i][0] - holes_center[1+i][0])*x_coff 
                            disty = (holes_center[0+i][1] - holes_center[1+i][1])*y_coff
                            print("line length:",math.sqrt(distx*distx+disty*disty))
                            cv.line(src_print, holes_center[0+i], holes_center[i+1], (255,100,255), 1)
                else:
                    ser.testing_func(0)

            elif state == state_dict['BLACK_BOX_DETECTION']:  
                thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)[1] # change the image to binary image by setting a threshold 117, when change new lighting setup you should change the 2nd parameter
                cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                for c in cnts: 
                    area = cv.contourArea(c) 
                    if(area > 5000 and area < 20000):# the hour and min hands
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
                        cv.drawContours(src_print,[box],0,(255,0,0),1)
                    # approx = cv.approxPolyDP(c, 0.1*cv.arcLength(c, True), True)
                    # cv.polylines(src_print, [approx], True, (255, 0, 0), 2)
                    # if len(approx) == 4:
                    #     x, y, w, h = cv.boundingRect(c)

                

        cv.imshow('frame', binaryIMG)
        cv.imshow('thresh', thresh) 
        cv.imshow('src', src_print)
        cv.setMouseCallback('src', click_event)
        print('q:', ord('q'))
        print('Q:', ord('Q'))
        print('idk: ', 0xFF)
        if cv.waitKey(4) & 0xFF == ord('q'):
            break

    vid_cam.release()
    cv.destroyAllWindows()
    print('Hi')
