import torch
import cv2
import pyttsx3

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path="C:/Users/rahul/Downloads/Weapon_Detection--main_WORKING/yolov5n6.pt")

# Set device to GPU if available, else use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the class labels
class_labels = ['motorcycle']
engine = pyttsx3.init()
# Function to perform two-wheeler detection on a video
video_path = "C:/Users/rahul/Downloads/4.mp4"
#4,22(engine)
video = cv2.VideoCapture(video_path)

while True:
    # Read the next frame
    ret, frame = video.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Filter out two-wheelers
    two_wheelers = results.pandas().xyxy[0]
    two_wheelers = two_wheelers[two_wheelers['name'].isin(class_labels)]
    # two_wheelers = int(two_wheelers)
    # Draw bounding boxes and labels on the frame
    for _, detection in two_wheelers.iterrows():
        xmin, ymin, xmax, ymax, _, confidence, class_id = detection
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        # label = class_labels[int(class_id)]
        label = class_id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}', (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # if(len(two_wheelers) > 3):
        #      engine.say("Abnormal Activity Detected!")
        #      engine.runAndWait()
       

    # Display the frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


