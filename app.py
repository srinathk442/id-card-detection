import streamlit as st
import cv2
import pandas as pd
import math
import easyocr
import re
from ultralytics import YOLO
import threading

# Load YOLO models
model_shorts = YOLO(r"model\shorts.pt")
model_idcard = YOLO(r"model\id_card.pt")

# Load OCR reader
reader = easyocr.Reader(['en'])

# Define the regular expression pattern for extracting text
pattern = re.compile("[0-9][0-9][A-Z]{3}[0-9]{4}")

# Functions for detecting objects
def id_card_detect(frame, box):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

    # Display class name above bounding box
    org = (x1, y1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2

    cv2.putText(frame, "id card", org, font, fontScale, color, thickness)

    # Display text extraction results
    

def shorts_detect(frame, box):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

    # Display class name above bounding box
    org = (x1, y1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2

    cv2.putText(frame, "shorts", org, font, fontScale, color, thickness)

    # Display text extraction results
    

def ocr_operation(frame):
    # Save the image
    cv2.imwrite('test.jpg', frame)

    # Perform OCR on the image
    results = reader.readtext('test.jpg')

    for result in results:
        text = result[1]  # Extract the OCR text from the result
        match = pattern.search(text)  # Search for the pattern in the text
        if match:
            st.write("Text extracted:", match.group()) 
            txt = match.group()
            if txt in ['22BAI1141','22BAI1120','22BAI1400','2BAI1387','22BAI1407']:
                st.success('Student is in the Database')
            else:
                st.error('Student not in database')

# Streamlit UI
st.title('Object Detection with YOLO and Text Extraction with OCR')

streaming = st.empty()

# Button to start webcam stream with a unique key
if streaming.button('Start Webcam Stream', key='start_stream'):
    stop_stream = st.button('Stop Stream', key='stop_stream_1')  # Create the 'Stop Stream' button outside the while loop
    try:
        cap = cv2.VideoCapture(0)
        while True:
            if stop_stream:  # Check for user input to stop the stream
                break
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            results_shorts = model_shorts(frame)
            results_id_card = model_idcard(frame)
            ocr_operation(frame)

            for result in results_shorts:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    shorts_detect(frame, box)

            for result in results_id_card:
                boxes = result.boxes
                for box in boxes:
                    id_card_detect(frame, box)

            # Display the frame
            streaming.image(frame, channels="BGR")

    except Exception as e:
        st.error(f"An error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
