import cv2
import numpy as np

# Load the video stream
cap = cv2.VideoCapture("C:/Users/rahul/Downloads/3.mp4")

# Define the lower and upper bounds of the fire color in the HSV space
fire_lower = np.array([0, 15, 15])
fire_upper = np.array([20, 255, 255])

# Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
zoom_factor = 2

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    height, width, _ = frame.shape
    
    # Apply a Gaussian blur to the frame
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    
    # Compute the absolute difference between the current frame and the previous frame
    diff = cv2.absdiff(bg_subtractor.apply(blur), 0)
    
    # Threshold the difference image
    threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Apply erosion and dilation to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(threshold, kernel, iterations=3)
    dilated = cv2.dilate(eroded, kernel, iterations=3)
    
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Mask the HSV image to extract the pixels in the fire color range
    mask = cv2.inRange(hsv, fire_lower, fire_upper)
    
    # Count the number of non-zero pixels in the masked image
    fire_pixels = cv2.countNonZero(mask)
    
    # Compute the percentage of fire pixels in the thresholded image
    total_pixels = dilated.shape[0] * dilated.shape[1]
    fire_percentage = (fire_pixels / total_pixels * 100)
    
    # Display the original frame with the fire percentage
    cv2.putText(frame, f'Fire percentage: {fire_percentage:.2f}%', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Fire detection', frame)
    
    # Check for keyboard input
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np

# # Load the video stream
# cap = cv2.VideoCapture(0)

# # Define the lower and upper bounds of the fire color in the HSV space
# fire_lower = np.array([0, 200, 200])
# fire_upper = np.array([20, 255, 255])

# # Initialize the background subtractor
# bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# while True:
#     # Read a frame from the video stream
#     ret, frame = cap.read()
    
#     if not ret:
#         print("Error reading video stream")
#         break

#     # Apply a Gaussian blur to the frame
#     blur = cv2.GaussianBlur(frame, (21, 21), 0)

#     # Convert the blurred frame to the HSV color space
#     hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

#     # Mask the HSV image to extract the pixels in the fire color range
#     mask = cv2.inRange(hsv, fire_lower, fire_upper)

#     # Apply erosion and dilation to the mask
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     eroded = cv2.erode(mask, kernel, iterations=2)
#     dilated = cv2.dilate(eroded, kernel, iterations=2)

#     # Compute the percentage of fire pixels in the thresholded image
#     fire_pixels = cv2.countNonZero(dilated)
#     total_pixels = dilated.shape[0] * dilated.shape[1]
#     fire_percentage = fire_pixels / total_pixels * 100

#     # Display the thresholded image and the percentage of fire pixels
#     cv2.imshow('Fire detection', dilated)
#     print(f'Fire percentage: {fire_percentage:.2f}%')

#     # Check for keyboard input
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and destroy all windows
# cap.release()
# cv2.destroyAllWindows()


