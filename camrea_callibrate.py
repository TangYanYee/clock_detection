import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:14.7:2.1,0:20.25:2.25].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

x_coff = 297/358
y_coff = 210/248
vid_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
src_points = np.array([[177., 350.], [396., 347.], [180., 200.], [395., 199.]], dtype = "float32")
dst_points = np.array([[177., 350.], [396., 350.], [177., 199.], [396., 199.]], dtype = "float32")

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
 
mtx = [[309.42268096,0.0,344.47160243], [0.0,305.46903569,223.85446847],[0.0,0.0,1.0]]
dist = [ 0.08619692,-0.04382931,-0.00451442,0.00761965,0.01002648]
while(vid_cam.isOpened()):
    img = vid_cam.read()[1]
    if img is not None:
        
        h,  w = img.shape[:2]
        print(h,'   ',w)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(np.float32(mtx), np.float32(dist), (w,h), 1, (w,h))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # undistort
        dst = cv2.undistort(img, np.float32(mtx), np.float32(dist), None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        img = dst[y:y+h, x:x+w]
        perspective = cv2.warpPerspective(img, M, (450,400 ), cv2.INTER_LINEAR)
                # Find the chess board corners
        # ret, corners = cv2.findChessboardCorners(gray, (7,9),
        #             cv2.CALIB_CB_ADAPTIVE_THRESH
        #             + cv2.CALIB_CB_FAST_CHECK +
        #             cv2.CALIB_CB_NORMALIZE_IMAGE)
        # # If found, add object points, image points (after refining them)
        # if ret == True:
        #     corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #     imgpoints.append(corners2)
        #     objpoints.append(objp)
        #     # Draw and display the corners
        #     cv2.drawChessboardCorners(img, (7,9), corners2,ret)
        #     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        #     # cam_mtx.append(mtx)
        #     # cam_dist.append(dist)
        #     print(mtx)
        #     print(dist)
    
        cv2.imshow('src', img)
        cv2.imshow('hah', perspective)
        cv2.setMouseCallback('hah', click_event)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()