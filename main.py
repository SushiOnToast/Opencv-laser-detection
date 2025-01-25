import cv2
import numpy as np
import socket

# Define the UDP server address and port (match the ESP32 code)
UDP_IP = "192.168.4.1"  # ESP32 AP IP address
UDP_PORT = 1234

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

stream_url = "http://192.168.4.1/stream"
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not connect to the stream!")
    exit()

# Trackbars for dynamic HSV adjustment
def nothing(x):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("LH", "Trackbars", 2, 179, nothing)
cv2.createTrackbar("LS", "Trackbars", 33, 255, nothing)
cv2.createTrackbar("LV", "Trackbars", 153, 255, nothing)
cv2.createTrackbar("UH", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("US", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("UV", "Trackbars", 255, 255, nothing)

kernel = np.ones((5, 5), np.uint8)

# Action determination function
def determine_action(center_x, center_y, frame_width, frame_height):
    center_frame_x = frame_width // 2
    center_frame_y = frame_height // 2
    threshold = 50  # Sensitivity threshold (adjustable)

    # Determine actions based on laser position
    if abs(center_x - center_frame_x) < threshold and abs(center_y - center_frame_y) < threshold:
        return "stop"
    elif center_y < center_frame_y - threshold:
        return "move forward"
    elif center_y > center_frame_y + threshold:
        return "move backward"
    elif center_x < center_frame_x - threshold:
        return "turn left"
    elif center_x > center_frame_x + threshold:
        return "turn right"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to fetch frame!")
        break

    frame_height, frame_width, _ = frame.shape

    # Adjust brightness/contrast
    frame = cv2.convertScaleAbs(frame)

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get dynamic HSV range
    lower_laser = np.array([
        cv2.getTrackbarPos("LH", "Trackbars"),
        cv2.getTrackbarPos("LS", "Trackbars"),
        cv2.getTrackbarPos("LV", "Trackbars"),
    ])
    upper_laser = np.array([
        cv2.getTrackbarPos("UH", "Trackbars"),
        cv2.getTrackbarPos("US", "Trackbars"),
        cv2.getTrackbarPos("UV", "Trackbars"),
    ])

    # Laser dot detection
    mask = cv2.inRange(hsv, lower_laser, upper_laser)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Sort contours by area (largest first)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) >= 20:  # Ignore very small reflections
            x, y, w, h = cv2.boundingRect(largest_contour)
            center_x, center_y = x + w // 2, y + h // 2

            # Draw rectangle and center circle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)

            # Determine action based on coordinates
            action = determine_action(center_x, center_y, frame_width, frame_height)
            print(f"Laser dot detected at: ({center_x}, {center_y}), Action: {action}")

            # Send action as UDP message to ESP32
            sock.sendto(action.encode(), (UDP_IP, UDP_PORT))

    cv2.imshow("ESP32-CAM Stream", frame)
    cv2.imshow("Laser Dot Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
