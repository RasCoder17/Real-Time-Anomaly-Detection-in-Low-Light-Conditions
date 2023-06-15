import cv2
import numpy as np
import pyttsx3
#import face_recognition
from datetime import datetime
import os 

path = "C:/Users/rahul/Downloads/Weapon_Detection--main_WORKING/Imageknown"
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 #       encode = face_recognition.face_encodings(img)[0]
  #      encodeList.append(encode)
    return encodeList  

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H : %M : %S \n')
            f.writelines(f'{name},{dtString}')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Load Yolo
# net = cv2.dnn.readNet("C:/Users/rahul/Downloads/Weapon_Detection--main_WORKING/yolov5_training_2000.weights", "C:/Users/rahul/Downloads/Weapon_Detection--main_WORKING/yolov5_testing.cfg")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# classes = ["Weapon"]

# Define the lower and upper bounds of the fire color in the HSV space
fire_lower = np.array([0, 15, 15])
fire_upper = np.array([20, 255, 255])

# Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Load the face cascade classifier
# face_cascade = cv2.CascadeClassifier('C:/Users/rahul/Downloads/Weapon_Detection--main_WORKING/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture("C:/Users/rahul/Downloads/3.mp4") #input video
while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 480))
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
   # facesCurFrame = face_recognition.face_locations(imgS)
   # num_faces = len(facesCurFrame)
   # encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    #for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
     #   matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
      #  faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        # print(faceDis)
       # matchIndex = np.argmin(faceDis)
       # if faceDis[matchIndex] < 0.50:
        #    name = classNames[matchIndex].upper() 
         #   markAttendance(name)
       # else:
        #    name = 'Unknown'    
            #print(name)
        #y1,x2,y2,x1 = faceLoc
        #y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        #cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        #cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        #cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    height, width, channels = img.shape
     # Apply a Gaussian blur to the frame
    blur = cv2.GaussianBlur(img, (21, 21), 0)
    
    # Compute the absolute difference between the current frame and the previous frame
    diff = cv2.absdiff(bg_subtractor.apply(blur), 0)
    
    # Threshold the difference image
    threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Apply erosion and dilation to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(threshold, kernel, iterations=3)
    dilated = cv2.dilate(eroded, kernel, iterations=3)
    
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Mask the HSV image to extract the pixels in the fire color range
    mask = cv2.inRange(hsv, fire_lower, fire_upper)
    
    # Count the number of non-zero pixels in the masked image
    fire_pixels = cv2.countNonZero(mask)
    
    # Compute the percentage of fire pixels in the thresholded image
    total_pixels = dilated.shape[0] * dilated.shape[1]
    fire_percentage = fire_pixels / total_pixels * 100

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detecting objects
    # blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # net.setInput(blob)
    # Perform face detection
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    #initiallizing the speech engine
    engine = pyttsx3.init()
    # layer_names = net.getLayerNames()

    # output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # colors = np.random.uniform(0, 255, size=(len(classes), 3))
    # outs = net.forward(output_layers)
    scale_factor = 0.8
    # Showing information on the screen
    # for (x, y, w, h) in faces:
    #     # w = w * scale_factor
    #     # h = h * scale_factor
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # class_ids = []
    # confidences = []
    # boxes = []
    # for out in outs:
    #     for detection in out:
    #         scores = detection[5:]
    #         class_id = np.argmax(scores)
    #         confidence = scores[class_id]
    #         if confidence > 0.4:
    #             # Object detected
    #             center_x = int(detection[0] * width)
    #             center_y = int(detection[1] * height)
    #             w = int(detection[2] * width * scale_factor)
    #             h = int(detection[3] * height * scale_factor)
    #             # Rectangle coordinates
    #             x = int(center_x - w / 2)
    #             y = int(center_y - h / 2)

    #             boxes.append([x, y, w, h])
    #             confidences.append(float(confidence))
    #             class_ids.append(class_id)

    # indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(indexes)
    #print("Number of faces: "+str(num_faces)) #Number of faces
    # if indexes == 0: 
    #     print("weapon detected in frame") 
    #     engine.say("Weapons Detected!")
    #     engine.runAndWait()
    # font = cv2.FONT_HERSHEY_PLAIN
    # for i in range(len(boxes)):
    #     if i in indexes:
    #         x, y, w, h = boxes[i]
    #         label = str(classes[class_ids[i]])
    #         color = colors[class_ids[i]]
    #         cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    #         cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    # Display the original frame with the fire percentage
    if(fire_percentage > 18):
        cv2.putText(img, f'Fire percentage: {fire_percentage:.2f}% ', (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print("fire detected in frame")         
        # engine.say("Fire Detected!")
        # engine.runAndWait()

    # frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
