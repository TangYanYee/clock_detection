import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:14.7:2.1,0:20.25:2.25].T.reshape(-1,2)
print(objp)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

x_coff = 297/358
y_coff = 210/248
vid_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vid_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
vid_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
# this need to click at 4 angle of a rectangle to calibrate the perspective transform
src_points = np.array([[598.*1.33333333333, 816.*1.33333333333], [1261.*1.33333333333, 816.*1.33333333333], [644.*1.33333333333, 396.*1.33333333333], [1231.*1.33333333333, 379.*1.33333333333]], dtype = "float32")
dst_points = np.array([[681.*1.33333333333, 911.*1.33333333333], [1337.*1.33333333333, 911.*1.33333333333], [681.*1.33333333333, 400.*1.33333333333], [1337.*1.33333333333, 400.*1.33333333333]], dtype = "float32")
M = cv2.getPerspectiveTransform(src_points, dst_points) 
def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x*x_coff, ' ', y*y_coff)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        # displaying the
        # on the Shell
        print(x*x_coff, ' ', y*y_coff)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
 
mtx = [[431.29886083,0.0,668.85105517], [0.0,427.29871171,369.08230931],[0.0,0.0,1.0]]
dist = [ 0.09036801,-0.0483161,-0.0032046,0.00379809 , 0.0113323]

mtx =  np.array([[5.17738780e+03,0.0,1.14821464e+03], [0.0,5.81785166e+03,6.87690122e+02],[0.0,0.0,1.0]], dtype = "float32")
dist =  np.array([ 3.42972020e+00,-5.24434430e+01,2.35853699e-02,8.89065380e-02 , 3.33197936e+02], dtype = "float32")
while(vid_cam.isOpened()):
    img = vid_cam.read()[1]

    if img is not None:
        img = cv2.warpPerspective(img, M, (2560, 1440), cv2.INTER_LINEAR)
        
        h,  w = img.shape[:2]
        print(h,'   ',w)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # uncomment the following part for camera callibration(L74~L98)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,9),
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_FAST_CHECK +
                    cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            print(corners2)
            objpoints.append(objp)
            print(objp)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,9), corners2,ret)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            print('mtx:')
            print(mtx)
            print('dist:')
            print(dist)
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.float32(mtx), np.float32(dist), (w,h), 1, (w,h))
            # undistort
            dst = cv2.undistort(img, np.float32(mtx), np.float32(dist), None, newcameramtx)
            # crop the image
            print('newcameramtx:')
            print(newcameramtx)
            print('roi:')
            print(roi)
            # x, y, w, h = roi
            # img = dst[y:y+h, x:x+w]
        img = cv2.resize(img, (1920,1080), interpolation = cv2.INTER_AREA)
        cv2.imshow('src', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
