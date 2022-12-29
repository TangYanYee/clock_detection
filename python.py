import sys
import cv2 as cv
import numpy as np
import math
import os
import pytesseract
from easyocr import Reader
pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract.exe'
# reader = Reader(['en'],gpu = False)
state_dict = {'FIND_CIRCLE': 0, 'DETECTING_NUMBER':1, 'DETECT_HANDS':2, 'DETECT_HOLES':3}
state = 3
from imutils.object_detection import non_max_suppression
vid_cam = cv.VideoCapture(0, cv.CAP_DSHOW)
center = (0,0)
small_center = (0,0)
dots_center = []
is_12_on_top = True
x_coff = 297/358
y_coff = 210/248
src_points = np.array([[177., 350.], [396., 347.], [180., 200.], [395., 199.]], dtype = "float32")
dst_points = np.array([[177., 350.], [396., 350.], [177., 199.], [396., 199.]], dtype = "float32")
mtx =  np.array([[309.42268096,0.0,344.47160243], [0.0,305.46903569,223.85446847],[0.0,0.0,1.0]], dtype = "float32")
cam_dist =  np.array([ 0.08619692,-0.04382931,-0.00451442,0.00761965,0.01002648], dtype = "float32")

newcameramtx = np.array([[323.44592,0.,349.27075],[0.,315.38824,221.84938],[0.,0.,1.]], dtype = "float32")
roi = (5, 9, 628, 462)
M = cv.getPerspectiveTransform(src_points, dst_points)
def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img
if __name__=='__main__':
    while(vid_cam.isOpened()):
        ret, src = vid_cam.read()
        if src is not None:
            # some filter to the image that need to execute every loop            
            gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
            dst = cv.undistort(src, mtx, cam_dist, None, newcameramtx)
            # crop the image
            roix, roiy, w, h = roi
            src = dst[roiy:roiy+h, roix:roix+w]
            src = cv.warpPerspective(src, M, (640 ,450), cv.INTER_LINEAR)
            src_print = src.copy()
            
            blurred = cv.GaussianBlur(gray, (7,7), 0) 
            thresh = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
            binaryIMG = cv.Canny(gray, 127, 255)
            gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY) #convert to gray scale
            blurred = cv.GaussianBlur(gray, (5,5), 0)  #blur the image
            thresh = cv.threshold(blurred, 117, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1] # change the image to binary image by setting a threshold 117, when change new lighting setup you should change the 2nd parameter
           
            binaryIMG = cv.Canny(gray, 127, 255) # add a canny edge detector

            
            #get size of the image
            rows = gray.shape[0] 
            cols = gray.shape[1]
            if state == state_dict['FIND_CIRCLE']:
                # Find circle
                circles = cv.HoughCircles(binaryIMG, cv.HOUGH_GRADIENT, 1, rows / 8,
                                        param1=255, param2=30,
                                        minRadius=130, maxRadius=160) #should adjust the min and max radius when the height of the camera changed
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        # circle center
                        center = (i[0], i[1])
                        cv.circle(src_print, center, 1, (0, 100, 100), 3)
                        radius = i[2]
                        # circle outline
                        cv.circle(src_print, center, radius, (255, 0, 255), 3)
                        state += 1 # go to next step
                
            elif state == state_dict['DETECTING_NUMBER']:  
                circles = cv.HoughCircles(binaryIMG, cv.HOUGH_GRADIENT, 1, rows / 8,
                                          param1=255, param2=30,
                                          minRadius=130, maxRadius=160) #should adjust the min and max radius when the height of the camera changed
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        center = (i[0], i[1])
                # Find small dots 
                cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) #find small dots by finding contours
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                dots_center = []
                for c in cnts: 
                    area = cv.contourArea(c) 
                    #find the hole of the clock
                    if area > 23 and area < 120: 
                        ((x,y), r) = cv.minEnclosingCircle(c) 
                        dist = math.dist(center, [x,y])
                        if(dist<20):
                            small_center = [int(x),int(y)]
                            print("small_center_x:",x,"small_center_y:",y)
                            cv.circle(src_print, small_center, int(r), (0, 255, 255), 2) 
                            cv.circle(src_print, small_center, 75, (0, 255, 255), 2) 
                            cv.circle(src_print, small_center, 95, (0, 255, 255), 2) 
                    #find the small dots of the clock
                    elif area > 3 and area < 20: 
                        ((x, y), r) = cv.minEnclosingCircle(c) 
                        #fliter the useful dots by the distance between the small center and dots center
                        dist = math.dist(small_center, [x,y])
                        if(dist>75 and dist<95):
                            ix = int(x)
                            iy = int(y)
                            up_text = ''
                            down_text = ''
                            #filter out the up and down dots by the difference of the x cooridinate
                            if(abs(ix - small_center[0])<10):
                                dots_center.append([ix,iy])
                                if(iy - small_center[1]<0):
                                    cv.rectangle(src_print, (ix-30, iy),(ix+30, iy-45), (36, 255, 12), 1)
                                    img = src[iy-45:iy,ix-30:ix+30]
                                    cv.imshow("up",img)
                                    up_text = pytesseract.image_to_string(img, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789') #use py tesseract for the text detection
                                    print("up: ",up_text, '.')
                                else:
                                    cv.rectangle(src_print, (ix-30, iy),(ix+30, iy+45), (36, 255, 12), 1)
                                    img = src[iy:iy+45,ix-30:ix+30]
                                    down_text = pytesseract.image_to_string(img, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789') #use py tesseract for the text detection
                                    print("down: ", down_text, '.')
                                    cv.imshow("down",img)
                                print("x:",ix,"y:",iy)
                                print("area:",area)
                                cv.circle(src_print, (ix, iy), int(r), (36, 255, 12), 2)
                                #determin the clock orientation by where is 12, notice that when the clock is upside down, the 6 will be detected as 9
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
                center_of_mass = 0
                print("clock ang: ", math.atan2(dots_center[0][1]-dots_center[1][1], dots_center[0][0]-dots_center[1][0])/np.pi*180.0)
                #should change the below part if the position for putting the hour/min/second hand is changed
                startpointx = 500
                startpointy = 100
                endpointx = cols
                endpointy = rows-150
                img = thresh[startpointy:endpointy,startpointx:endpointx] #cut out part of the threshold inamge
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
                cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) 
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                circle_center = []
                circles = cv.HoughCircles(binaryIMG, cv.HOUGH_GRADIENT, 1, rows / 8,
                                        param1=255, param2=30,
                                        minRadius=160, maxRadius=190) #should adjust the min and max radius when the height of the camera changed
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        # circle center
                        center = (int(i[0]), int(i[1]))
                        cv.circle(src_print, center, 1, (0, 100, 100), 3)
                        radius = int(i[2])
                        # circle outline
                        print(radius)
                        cv.circle(src_print, center, radius, (255, 0, 255), 3)
                    for c in cnts: 
                        area = cv.contourArea(c) 
                        if(area > 30 and area < 100):# the hour and min hands
                            ((x, y), r) = cv.minEnclosingCircle(c) 
                            # fliter the useful dots by the distance between the small center and dots center
                            dist = math.dist(center, [x,y])
                            if(dist>160 and dist<185):
                                circle_center.append([int(x),int(y)])
                                cv.circle(src_print,(int(x),int(y)),int(r),(255,255,0),1)
                if len(circle_center) == 6:
                    for i in range(0,6,2):
                        distx = (circle_center[0+i][0] - circle_center[1+i][0])*x_coff
                        disty = (circle_center[0+i][1] - circle_center[1+i][1])*y_coff
                        print("line length:",math.sqrt(distx*distx+disty*disty))
                        cv.line(src_print,circle_center[0+i],circle_center[i+1],(255,0,255),1)

        cv.imshow('frame', binaryIMG)
        cv.imshow('thresh', thresh) 
        cv.imshow('src', src_print)
        if cv.waitKey(100) & 0xFF == ord('q'):
            break

    vid_cam.release()
    cv.destroyAllWindows()
    print('Hi')
