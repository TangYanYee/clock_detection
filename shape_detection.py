import cv2 as cv
import numpy as np
import math
import pytesseract

pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract.exe'
# this need to click at 4 angle of a rectangle to calibrate the perspective transform
src_points = np.array([[160., 684.], [990., 677.], [199., 134.], [985., 138.]], dtype = "float32")
dst_points = np.array([[160., 684.], [990., 684.], [160., 95.], [990., 95.]], dtype = "float32")

# this is camera matrix and dist, whenever we use a new camera we need calibrate this by using the camera_calibrate.py after getting the result just replace the value

mtx =  np.array([[5.17738780e+03,0.0,1.14821464e+03], [0.0,5.81785166e+03,6.87690122e+02],[0.0,0.0,1.0]], dtype = "float32")
cam_dist =  np.array([ 3.42972020e+00,-5.24434430e+01,2.35853699e-02,8.89065380e-02 , 3.33197936e+02], dtype = "float32")
# this is output of the getOptimalNewCameraMatrix, whenever we use a new camera we need to calibrate this by using the camera_calibrate.py after getting the result
newcameramtx = np.array([[5.4086167e+03,0.,1.1758104e+03],[0., 5.9403320e+03,6.8499677e+02],[0.,0.,1.]], dtype = "float32")
roi = (7, 33, 1899, 1019)
roix, roiy, w, h = roi

M = cv.getPerspectiveTransform(src_points, dst_points) 

class hands_data():
    def __init__(self, angle, length):
        self.angle = angle
        self.length = length
    def show_data(self):
        print('ang:', self.angle)
        print('len:', self.length)

def filteringImage(src):
    # some filter to the image that need to execute every loop            
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # counter the fish eye effect of the camera
    dst = cv.undistort(src, mtx, cam_dist, None, newcameramtx)
    cv.imshow('hhhh', dst)
    # crop the image
    src = dst[roiy:roiy+h, roix:roix+w]
    src = cv.warpPerspective(src, M, (1920, 1080), cv.INTER_LINEAR)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY) #convert to gray scale
    blurred = cv.GaussianBlur(gray, (7, 7), 0) 
    thresh = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1] # THRESH_OTSU will auto choose the best value for threshold
    binaryIMG = cv.Canny(gray, 127, 255)
    return src, gray, thresh, binaryIMG

def find_cg(c):
    m = cv.moments(c)
    #find the center of mass to determine the orientation of the second hand, becuase the circle of the second hand is hard to detect
    if( m["m00"] != 0):
        x = int(m["m10"] / m["m00"])
        y = int(m["m01"] / m["m00"])
        center_of_mass = (int(x),int(y))
        return center_of_mass

def detect_circle(src, src_print, minR, maxR, votes = 30, circle_xy = None, mindis = 0, maxdis = 0):
    center_list = []
    
    #get size of the image
    rows = src.shape[0] 
    circles = cv.HoughCircles(src, cv.HOUGH_GRADIENT, 1, rows / 8, param1=255, param2 = votes, minRadius = minR, maxRadius = maxR) #should adjust the min and max radius when the height of the camera changed
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # circle center
            center = [i[0], i[1]]
            if(circle_xy is not None):
                dis_in_range = math.dist(center,circle_xy)>mindis and math.dist(center,circle_xy) < maxdis
            if((circle_xy is not None and dis_in_range) or circle_xy is None):
                radius = i[2]
                print(radius)
                center_list.append(center)
                cv.circle(src_print, center, radius, (255, 0, 255), 3)
    return center_list
def detect_orientation(src,src_print, dots_center, small_center):
    up_text = ''
    down_text = ''
    small_center = small_center[0]
    #filter out the up and down dots by the difference of the x cooridinate
    for c in dots_center:
        if(abs(c[0] - small_center[0])<20):
            if(c[1] - small_center[1]<0):
                cv.rectangle(src_print, (c[0]-50, c[1]-10),(c[0]+50, c[1]-120), (36, 255, 12), 1)
                img = src[c[1]-120:c[1]-10,c[0]-50:c[0]+50]
                cv.imshow("up",img)
                up_text = pytesseract.image_to_string(img, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789') #use py tesseract for the text detection
                print("up: ",up_text)
            else:
                cv.rectangle(src_print, (c[0]-50, c[1]+10),(c[0]+50, c[1]+120), (36, 255, 12), 1)
                img = src[c[1]+10:c[1]+120,c[0]-50:c[0]+50]
                down_text = pytesseract.image_to_string(img, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789') #use py tesseract for the text detection
                print("down: ", down_text)
                cv.imshow("down",img)
            print("x:",c[0],"y:",c[1])
            #determine the clock orientation by where is 12, notice that when the clock is upside down, the 6 will be detected as 9
    if(up_text.find("12") != -1 or down_text.find("6") != -1):
        print("12 on top")
        return True
    elif(up_text.find("9") != -1 or down_text.find("12") != -1):
        print("12 on bottom")
        return False
    return None

def detect_dots(src,src_print, center_pos ,minA ,maxA ,minDis ,maxDis):
    cnts = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) #find small dots by finding contours
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    dots_center = []
    for c in cnts: 
        area = cv.contourArea(c) 
        #find the hole of the clock
        if area > minA and area < maxA: 
            (x,y),r = cv.minEnclosingCircle(c)
            print("small center2", center_pos[0])
            dist = math.dist(center_pos[0], [x,y])
            if(dist>minDis and dist< maxDis):
                cv.circle(src_print, (int(x),int(y)), int(r), (255, 255, 100), 1)  
                dots_center.append([int(x),int(y)])
    return dots_center
def detect_hand(src, src_print):
    arr = []
    center_of_mass = 0
    color = 0
    # #find contours of 3 hands
    cnts = cv.findContours(src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
    cv.drawContours(src_print, cnts[0], -1, (255, 255, 255), 2)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:

        approx = cv.approxPolyDP(c, 50, True)
        if len(approx) == 2:
            color += 1
            # print(approx)
            len_hand = math.dist(approx[0][0],approx[1][0])
            center_of_mass = find_cg(c)
            if(center_of_mass is not None):
                cv.circle(src_print, center_of_mass, 1, (color * 70, 100, 100), 3)
                if(math.dist(center_of_mass,approx[0][0]) < math.dist(center_of_mass,approx[1][0])):
                    approx[[0,1]] = approx[[1,0]]
                cv.circle(src_print, approx[0][0], 1, (color * 70, 100, 0), 3)
            cv.line(src_print, approx[0][0],approx[1][0], (color * 70, 0, 0), 2)
            try: 
                rows,cols = src.shape[:2]
                [vx,vy,x,y] = cv.fitLine(c, cv.DIST_L2,0,0.01,0.01)
                # print(vx,vy,x,y)
                lefty = int((-x*vy/vx) + y)
                righty = int(((cols-x)*vy/vx)+y)
                print(vy/vx)
                cv.line(src_print,(int(x),int(y)),(int(approx[0][0][0]),int(approx[0][0][0]*vy/vx + lefty)),(0,255,0),2)
                ang = math.atan2(approx[0][0][0]*vy/vx + lefty - y,approx[0][0][0] - x)/np.pi*180.0 +90.0
                arr.append(hands_data(ang, len_hand))
            except:
                print('Some error')
            cv.circle(src_print, [int(x),int(y)], 1, (0, 0, 255), 3)
    return arr
