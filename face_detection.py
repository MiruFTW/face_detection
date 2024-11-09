import cv2
import os
from deepface import DeepFace  # Import deepface for emotion detection

# Load the pre-trained face detection model
cascPath = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Start capturing video
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around each face and display position data
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract the face region for emotion analysis
        face_img = frames[y:y + h, x:x + w]

        # Detect emotion with deepface
        try:
            result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            
            # Check if result is a list and get the first item if it is
            if isinstance(result, list):
                result = result[0]  # Access the first face result if a list is returned
            
            emotion = result['dominant_emotion']  # Access the dominant emotion
            position_text = f"Emotion: {emotion} | Position: ({x}, {y})"
        except Exception as e:
            position_text = f"Position: ({x}, {y}) | Error: {str(e)}"

        # Display emotion and face position data
        cv2.putText(frames, position_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Display the resulting frame
    cv2.imshow('Video', frames)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the display window
video_capture.release()
cv2.destroyAllWindows()
