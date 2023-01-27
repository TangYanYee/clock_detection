import torch
import cv2

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:\\Users\\User\\Desktop\\clock_detection\\best.pt')
arr = ['clock','outer_box','clock_movement','hour','sec','min','12','6','frame']
# Image
vid_cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
vid_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
vid_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
while(vid_cam.isOpened()):
    img = vid_cam.read()[1]
    if img is not None:

        # Inference
        results = model(img)

        results.pandas().xyxy[0]
        for box in results.xyxy[0]: 
            print(box[5])
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])
            cv2.putText(img, arr[int(box[5])], (xA, yA), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.imshow('src', img)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        
vid_cam.release()