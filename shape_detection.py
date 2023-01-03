import cv2 as cv
import numpy as np
import math
import pytesseract
pytesseract.pytesseract.tesseract_cmd='C:/Program Files/Tesseract-OCR/tesseract.exe'
# this need to click at 4 angle of a rectangle to calibrate the perspective transform
src_points = np.array([[130., 357.], [487., 360.], [143., 110.], [483., 119.]], dtype = "float32")
dst_points = np.array([[130., 357.], [487., 357.], [130., 110.], [487., 110.]], dtype = "float32")

# this is camera matrix and dist, whenever we use a new camera we need calibrate this by using the camera_calibrate.py after getting the result just replace the value
mtx =  np.array([[309.42268096,0.0,344.47160243], [0.0,305.46903569,223.85446847],[0.0,0.0,1.0]], dtype = "float32")
cam_dist =  np.array([ 0.08619692,-0.04382931,-0.00451442,0.00761965,0.01002648], dtype = "float32")
# this is output of the getOptimalNewCameraMatrix, whenever we use a new camera we need to calibrate this by using the camera_calibrate.py after gettubg the result
newcameramtx = np.array([[323.44592,0.,349.27075],[0.,315.38824,221.84938],[0.,0.,1.]], dtype = "float32")
roi = (5, 9, 628, 462)
roix, roiy, w, h = roi

M = cv.getPerspectiveTransform(src_points, dst_points) # get perspective transform 


def filteringImage(src):
    # some filter to the image that need to execute every loop            
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # counter the fish eye effect of the camera
    dst = cv.undistort(src, mtx, cam_dist, None, newcameramtx)
    # crop the image
    src = dst[roiy:roiy+h, roix:roix+w]
    src = cv.warpPerspective(src, M, (640, 450), cv.INTER_LINEAR)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY) #convert to gray scale
    blurred = cv.GaussianBlur(gray, (7, 7), 0) 
    thresh = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1] # THRESH_OTSU will auto choose the best value for threshold
    binaryIMG = cv.Canny(gray, 127, 255)
    return src, gray, thresh, binaryIMG

def detect_circle(src, src_print, minR, maxR, votes = 30):
    center_list = []
    
    #get size of the image
    rows = src.shape[0] 
    circles = cv.HoughCircles(src, cv.HOUGH_GRADIENT, 1, rows / 8, param1=255, param2 = votes, minRadius = minR, maxRadius = maxR) #should adjust the min and max radius when the height of the camera changed
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # circle center
            center = [i[0], i[1]]
            radius = i[2]
            center_list.append(center)
            cv.circle(src_print, center, radius, (255, 0, 255), 3)
    return center_list
def orientation_detect(src,src_print, dots_center, small_center):
    up_text = ''
    down_text = ''
    small_center = small_center[0]
    #filter out the up and down dots by the difference of the x cooridinate
    for c in dots_center:
        if(abs(c[0] - small_center[0])<12):
            if(c[1] - small_center[1]<0):
                cv.rectangle(src_print, (c[0]-30, c[1]),(c[0]+30, c[1]-45), (36, 255, 12), 1)
                img = src[c[1]-45:c[1],c[0]-30:c[0]+30]
                cv.imshow("up",img)
                up_text = pytesseract.image_to_string(img, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789') #use py tesseract for the text detection
                print("up: ",up_text)
            else:
                cv.rectangle(src_print, (c[0]-30, c[1]),(c[0]+30, c[1]+45), (36, 255, 12), 1)
                img = src[c[1]:c[1]+45,c[0]-30:c[0]+30]
                down_text = pytesseract.image_to_string(img, lang='eng', config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789') #use py tesseract for the text detection
                print("down: ", down_text)
                cv.imshow("down",img)
            print("x:",c[0],"y:",c[1])
            #determin the clock orientation by where is 12, notice that when the clock is upside down, the 6 will be detected as 9
    if(up_text.find("12") != -1 or down_text.find("6") != -1):
        print("12 on top")
        return True
    elif(up_text.find("9") != -1 or down_text.find("12") != -1):
        print("12 on bottom")
        return False
    return None
def detect_dots(src,src_print, detect_type, *args):
    """args :[0]center_pos [1]minA [2]maxA [3]minDis [4]maxDis [5]func to call(if needed) """
    cnts = cv.findContours(src, detect_type, cv.CHAIN_APPROX_NONE) #find small dots by finding contours
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    dots_center = []
    for c in cnts: 
        area = cv.contourArea(c) 
        #find the hole of the clock
        for boundary in args:
            if area > boundary[1] and area < boundary[2]: 
                (x,y),r = cv.minEnclosingCircle(c)
                
                print("small center2", boundary[0][0])
                dist = math.dist(boundary[0][0], [x,y])
                if(dist> boundary[3] and dist< boundary[4]):
                    cv.circle(src_print, (int(x),int(y)), int(r), (255, 255, 100), 1)  
                    dots_center.append([int(x),int(y)])
                    if(len(boundary)>5):
                        boundary[5]()
    return dots_center
