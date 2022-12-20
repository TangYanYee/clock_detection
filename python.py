import sys
import cv2 as cv
import numpy as np
import math
import os
import pytesseract
from easyocr import Reader
pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract.exe'
# reader = Reader(['en'],gpu = False)
state_dict = {'FIND_CIRCLE': 0, 'DETECTING_NUMBER':1, 'DETECT_HANDS':2}
state = 0
from imutils.object_detection import non_max_suppression
vid_cam = cv.VideoCapture(0, cv.CAP_DSHOW)
center = (0,0)
small_center = (0,0)
dots_center = []
is_12_on_top = True
if __name__=='__main__':
    while(vid_cam.isOpened()):
        ret, src = vid_cam.read()
        src_print = src.copy()
        if src is not None:
            gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            
            blurred = cv.GaussianBlur(gray, (5,5), 0) 
            thresh = cv.threshold(blurred, 117, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
            binaryIMG = cv.Canny(gray, 127, 255)
            rows = gray.shape[0]
            cols = gray.shape[1]
            if state == state_dict['FIND_CIRCLE']:
                # Find circle
                circles = cv.HoughCircles(binaryIMG, cv.HOUGH_GRADIENT, 1, rows / 8,
                                        param1=255, param2=30,
                                        minRadius=130, maxRadius=160)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        # circle center
                        cv.circle(src_print, center, 1, (0, 100, 100), 3)
                        radius = i[2]
                        print(radius)
                        # circle outline
                        cv.circle(src_print, center, radius, (255, 0, 255), 3)
                        state += 1
                
            elif state == state_dict['DETECTING_NUMBER']:  
                circles = cv.HoughCircles(binaryIMG, cv.HOUGH_GRADIENT, 1, rows / 8,
                                        param1=255, param2=30,
                                        minRadius=130, maxRadius=160)
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        center = (i[0], i[1])
                # Find small dots 
                cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                dots_center = []
                for c in cnts: 
                    area = cv.contourArea(c) 
                    if area > 23 and area < 120: 
                        ((x,y), r) = cv.minEnclosingCircle(c) 
                        dist = math.dist(center, [x,y])
                        if(dist<20):
                            small_center = [int(x),int(y)]
                            print("small_center_x:",x,"small_center_y:",y)
                            cv.circle(src_print, small_center, int(r), (0, 255, 255), 2) 
                            cv.circle(src_print, small_center, 75, (0, 255, 255), 2) 
                            cv.circle(src_print, small_center, 95, (0, 255, 255), 2) 
                    elif area > 3 and area < 20: 
                        ((x, y), r) = cv.minEnclosingCircle(c) 
                        dist = math.dist(small_center, [x,y])
                        if(dist>75 and dist<95):
                            ix = int(x)
                            iy = int(y)
                            up_text = ''
                            down_text = ''
                            if(abs(ix - small_center[0])<10):
                                dots_center.append([ix,iy])
                                if(iy - small_center[1]<0):
                                    cv.rectangle(src_print, (ix-30, iy),(ix+30, iy-45), (36, 255, 12), 1)
                                    img = src[iy-45:iy,ix-30:ix+30]
                                    cv.imshow("up",img)
                                    up_text = pytesseract.image_to_string(img, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
                                    print("up: ",up_text, '.')
                                else:
                                    cv.rectangle(src_print, (ix-30, iy),(ix+30, iy+45), (36, 255, 12), 1)
                                    img = src[iy:iy+45,ix-30:ix+30]
                                    down_text = pytesseract.image_to_string(img, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
                                    print("down: ", down_text, '.')
                                    cv.imshow("down",img)
                                print("x:",ix,"y:",iy)
                                print("area:",area)
                                cv.circle(src_print, (ix, iy), int(r), (36, 255, 12), 2)
                            if(up_text.find("12") != -1 or down_text.find("6") != -1):
                                print("12 on top")
                                is_12_on_top = True
                            elif(up_text.find("9") != -1 or down_text.find("12") != -1):
                                print("12 on bottom")
                                is_12_on_top = False
                            if(len(dots_center) == 2):
                                state += 1
            elif state == state_dict['DETECT_HANDS']:
                # if is_12_on_top is trueign it with the top
                # if is_12_on_top is flase, align it with the bottom
                # crop the image
                center_of_mass = []
                print("clock ang: ", math.atan2(dots_center[0][1]-dots_center[1][1], dots_center[0][0]-dots_center[1][0])/np.pi*180.0)
                img = thresh[100:rows-150,500:cols]
                img2 = binaryIMG[100:rows-150,500:cols]
                img3 = np.zeros((rows-250, cols-500, 3), dtype = np.uint8)
                img3 = cv.cvtColor(img3,cv.COLOR_BGR2GRAY)
                cv.rectangle(src_print,(500,100),(cols,rows-150),(0,0,0),1)
                cnts = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
                cv.drawContours(src_print, cnts[0], -1, (255, 255, 255), 2)
                cv.drawContours(img3, cnts[0], -1, (255, 255, 255), 1)
                cv.imshow("img3", img3)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                hand_length = []
                hand_ang = []
                for c in cnts: 
                    area = cv.contourArea(c) 
                    if(area > 500):
                        rect = cv.minAreaRect(c)
                        box = cv.boxPoints(rect)
                        box = np.int0(box)
                        top_point_y = np.min(box[:, 1])
                        bottom_point_y = np.max(box[:, 1])
                        top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
                        bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
                        # Retrieve the key parameters of the rotated bounding box
                        width = int(rect[1][0])
                        height = int(rect[1][1])
                        if((width > 10 and width < 20) or (height > 10 and height < 20)):   
                            center = (int(rect[0][0]),int(rect[0][1])) 
                            hand_length.append(max(width,height))
                            if(width > height):
                                a = rect[2] - 90
                                hand_ang.append(a)
                            else:   
                                hand_ang.append(rect[2])
                            cv.drawContours(src_print,[box],0,(255,0,0),1)
                        circles = cv.HoughCircles(img3, cv.HOUGH_GRADIENT, 1, rows / 8,
                                                param1=255, param2=5,
                                                minRadius=4, maxRadius=7)
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for i in circles[0, :]:
                                center = (i[0], i[1])
                                radius = i[2]
                                if(math.dist((top_point_x,top_point_y),center)<12 or math.dist((bottom_point_x,bottom_point_y),center)<12):
                                    cv.circle(src_print, center, radius, (255, 0, 255), 2)    
                    else:
                        m = cv.moments(c)
                        if( m["m00"] != 0):
                            x = int(m["m10"] / m["m00"])
                            y = int(m["m01"] / m["m00"])
                            center_of_mass = (x,y)
                            cv.circle(src_print, (int(x),int(y)), 1, (255, 0, 255), 1)
                if len(hand_length)>0:
                    print("Min Hand ang: ",hand_ang[hand_length.index(max(hand_length))])
                    print("Hour Hand ang: ",hand_ang[hand_length.index(min(hand_length))])
                linesP = cv.HoughLinesP(img2, 1, np.pi / 180, 15, None, 125, 10)
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
                    if(center_of_mass is not None):
                        if(math.dist(center_of_mass,(l[0],l[1])) < math.dist(center_of_mass,(l[2],l[3]))):
                            l[0], l[2] = l[2], l[0]
                            l[1], l[3] = l[3], l[1]
                    cv.circle(src_print, (l[0],l[1]), 1, (255, 255, 50), 3)
                    print("Sec Hand ang: ", math.atan2(l[1]-l[3], l[0]-l[2])/np.pi*180.0)
                    print("point1: ",l[0],",",l[1],"|| point2: ",l[2],",",l[3])
                    cv.line(src_print, (l[0]+500, l[1]+100), (l[2]+500, l[3]+100), (0,0,255), 3, cv.LINE_AA)


        cv.imshow('frame', binaryIMG)
        cv.imshow('thresh', thresh) 
        cv.imshow('src', src_print)
        if cv.waitKey(100) & 0xFF == ord('q'):
            break

    vid_cam.release()
    cv.destroyAllWindows()
    print('Hi')
