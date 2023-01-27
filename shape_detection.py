import cv2 as cv
import numpy as np
import torch
import math
import pytesseract
import serial_comm as xyz

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\User\\Desktop\\clock_detection\\best.pt')
arr = ['clock','outer_box','clock_movement','hour','sec','min','12','6','frame']
# clock
# outer_box
# clock_movement
# hour
# sec
# min
# 12
# 6
# frame
pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract.exe'
# this need to click at 4 angle of a rectangle to calibrate the perspective transform
src_points = np.array([[717.*1.33333333333, 577.*1.33333333333], [1200.*1.33333333333, 577.*1.33333333333], [745.*1.33333333333, 284.*1.33333333333], [1184.*1.33333333333, 273.*1.33333333333]], dtype = "float32")
dst_points = np.array([[717.*1.33333333333, 577.*1.33333333333], [1200.*1.33333333333, 577.*1.33333333333], [717.*1.33333333333, 230.*1.33333333333], [1200.*1.33333333333, 230.*1.33333333333]], dtype = "float32")
M = cv.getPerspectiveTransform(src_points, dst_points) 

# this is camera matrix and dist, whenever we use a new camera we need calibrate this by using the camera_calibrate.py after getting the result just replace the value
mtx =  np.array([[1.00665648e+04, 0.00000000e+00, 1.30858595e+03], [0.0,9.95425408e+03, 7.08807330e+02],[0.0,0.0,1.0]], dtype = "float32")
cam_dist =  np.array([ 5.71346288e+00, -6.48707545e+02, -1.09636166e-02, -3.50020295e-02,2.21666555e+04], dtype = "float32")
# this is output of the getOptimalNewCameraMatrix, whenever we use a new camera we need to calibrate this by using the camera_calibrate.py after getting the result
newcameramtx = np.array([[1.0225396e+04, 0.00000000e+00,1.2914183e+03], [0.0,1.0023616e+04,7.0624719e+02],[0.,0.,1.]], dtype = "float32")
roi = (23, 24, 2517, 1391)
roix, roiy, w, h = roi

x_coff = 300/640 
offset = (581,168) #Actually is useless, just keep is as this value, it is better to tune the distance offset in serial_comm.py
def xy2xyz(coor, z):
    return [coor[0], coor[1], z]
def pixel2xy(coor):
    coor[1] = 1440 - coor[1]
    # coor[1] = 1080 - coor[1]
    coor[0] -= offset[0]
    coor[1] -= offset[1]
    coor[0] *= x_coff
    coor[1] *= x_coff
    print(xy2xyz(coor,0))
    # xyz.move(xy2xyz(coor,0))
    return coor

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        print(x*x_coff, ' ', y*x_coff)
    # checking for right mouse clicks    
    if event == cv.EVENT_RBUTTONDOWN:
        print(x, ' ', y)
    # checking for middle mouse clicks    
    if event == cv.EVENT_MBUTTONDOWN:
        print(pixel2xy([x,y]))


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
    # crop the image
    src = dst[roiy:roiy+h, roix:roix+w]
    src = cv.warpPerspective(src, M, (2560, 1440), cv.INTER_LINEAR)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY) #convert to gray scale
    blurred = cv.GaussianBlur(gray, (7, 7), 0) 
    thresh = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1] # THRESH_OTSU will auto choose the best value for threshold
    binaryIMG = cv.Canny(gray, 127, 255)
    return src, gray, thresh, binaryIMG
def filteringImage2(src):
    # some filter to the image that need to execute every loop         
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY) #convert to gray scale    
    thresh = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)[1]
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
            center = (i[0], i[1])
            if(circle_xy is not None):
                dis_in_range = math.dist(center,circle_xy)>mindis and math.dist(center,circle_xy) < maxdis
            if((circle_xy is not None and dis_in_range) or circle_xy is None):
                radius = i[2]
                print(radius)
                center_list.append(center)
                cv.circle(src_print, center, radius, (255, 0, 255), 3)
    return center_list
def detect_orientation(src,src_print, dots_center, small_center):
    point = [(0,0),(0,0)]
    small_center = small_center[0]
    results = model(src)#load result from model with src as source
    results.pandas().xyxy[0]#get the rectangles coordinates
    for box in results.xyxy[0]: 
        if(box[5] == 6 or box[5] == 7): #filter out which one is 6 and 12 by the id of box[5]
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])
            mid_pt = ((xA+xB)/2,(yA+yB)/2)
            cv.putText(src_print, arr[int(box[5])], (xA, yA), cv.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv.LINE_AA)
            cv.rectangle(src_print, (xA, yA), (xB, yB), (0, 255, 0), 2)
            #store the dots_center to point array
            for c in dots_center:
                if(math.dist(c, mid_pt) < 70):
                    cv.circle(src_print, (int(c[0]),int(c[1])), int(5), (0, 255, 0), 3)  
                    point[int(box[5])-6] = (int(c[0]),int(c[1]))
            
    return point
def detect_black_box(src,src_print):
    clock_movement = []
    outer_box = []
    results = model(src) #load result from model with src as source
    results.pandas().xyxy[0] #get the rectangles coordinates
    for box in results.xyxy[0]: 
        if(box[5] == 1 or box[5] == 2): #filter out which one is clockmovement and outerbox by the id of box[5]
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])
            print('Hi11')
            cv.putText(src_print, arr[int(box[5])], (xA, yA), cv.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 0, 0), 2, cv.LINE_AA)
            cv.rectangle(src_print, (xA, yA), (xB, yB), (0, 255, 0), 2)
            if(box[5] == 1):
                outer_box.extend([xA-30, yA-30, xB+30, yB+30])
            else:
                clock_movement.extend([xA-30, yA-30, xB+30, yB+30])
    return outer_box, clock_movement   
def detect_dots(src,src_print, center_pos ,minA ,maxA ,minDis ,maxDis):
    cnts = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_NONE) #find small dots by finding contours
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    dots_center = []
    for c in cnts: 
        area = cv.contourArea(c) 
        #find the hole of the clock
        if area > minA and area < maxA: 
            (x,y),r = cv.minEnclosingCircle(c)
            # print("small center2", center_pos[0])
            dist = math.dist(center_pos[0], [x,y])
            if(dist>minDis and dist< maxDis):
                cv.circle(src_print, (int(x),int(y)), int(r), (255, 255, 100), 1)  
                dots_center.append((int(x),int(y)))
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

        approx = cv.approxPolyDP(c, 50, True) # use approxPolyDP to estimate the clock hand as a straight line
        if len(approx) == 2:
            color += 1
            # print(approx)
            len_hand = math.dist(approx[0][0],approx[1][0])
            center_of_mass = find_cg(c) #find cg to find the orientaion of the hand
            if(center_of_mass is not None):
                cv.circle(src_print, center_of_mass, 1, (color * 70, 100, 100), 3)
                if(math.dist(center_of_mass,approx[0][0]) < math.dist(center_of_mass,approx[1][0])):
                    approx[[0,1]] = approx[[1,0]]
                cv.circle(src_print, (approx[0][0][0],approx[0][0][1]), 1, (color * 70, 100, 0), 3)
            cv.line(src_print, (approx[0][0][0],approx[0][0][1]),(approx[1][0][0],approx[1][0][1]), (color * 70, 0, 0), 2)
            try: 
                rows,cols = src.shape[:2]
                [vx,vy,x,y] = cv.fitLine(c, cv.DIST_L2,0,0.01,0.01)
                # print(vx,vy,x,y)
                
                lefty = int((-x*vy/vx) + y)
                righty = int(((cols-x)*vy/vx)+y)
                print(vy/vx)
                cv.line(src_print,(int(x),int(y)),(int(approx[0][0][0]),int(approx[0][0][0]*vy/vx + lefty)),(0,255,0),2)
                ang = math.atan2(approx[0][0][0]*vy/vx + lefty - y,approx[0][0][0] - x)/np.pi*180.0 +90.0 # y=mx+c
                arr.append(hands_data(ang, len_hand))
            except:
                print('Some error')
            cv.circle(src_print, (int(x),int(y)), 1, (0, 0, 255), 3)
    return arr
if __name__=='__main__':
    vid_cam = cv.VideoCapture(1, cv.CAP_DSHOW)
    vid_cam.set(cv.CAP_PROP_FRAME_WIDTH, 2560)#1920 1.3333333
    vid_cam.set(cv.CAP_PROP_FRAME_HEIGHT, 1440)#1080
    num = 0
    while(vid_cam.isOpened()):
        ret, src = vid_cam.read()
        if src is not None:
            src = filteringImage(src)[0]
            
            src = cv.resize(src, (1920,1080), interpolation = cv.INTER_AREA)
            cv.circle(src, (int(1920/2),int(1080/2)), 219+11, (100,100,255),1)
            cv.imshow('src', src)
            cv.setMouseCallback('src', click_event)
            key = cv.waitKey(4)
            if (key & 0xFF == ord('q')) or (key & 0xFF == ord('Q')):
                break
    vid_cam.release()
    cv.destroyAllWindows()
    print('Hi')
