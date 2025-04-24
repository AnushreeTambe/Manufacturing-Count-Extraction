
######################################################################################################################
##################################################################################################################
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

import streamlit as st
import os
from datetime import datetime
import smtplib
from email.message import EmailMessage

import pytesseract

# Explicitly set the Tesseract-OCR executable path
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Load YOLOv8 model for number detection
MODEL_PATH = "C:/Users/dell/Downloads/best (2).pt"  # Replace with your trained YOLOv8 model path
model = YOLO(MODEL_PATH)

# Define Paths for Data Storage
DATA_FOLDER = "extracted_data"
os.makedirs(DATA_FOLDER, exist_ok=True)
CSV_FILE_PATH = os.path.join(DATA_FOLDER, "extracted_numbers.csv")

# Define machine names globally
machine_names_5 = [
    "110 T Press", "160 T Press", "150 T Press", "250 T Press", "315 T Press"
]

machine_names_21 = [
    "100T Press", "80T Press", "60T Press", "45T Press", "45T Press",
    "45T Press", "60T Press", "75T Press", "100T Press", "75T Press",
    "150T Press", "250T Press", "80T Press", "45T Press", "45T Press",
    "63T Press", "63T Press", "45T Press", "45T Press", "63T Press", "63T Press"
]

# Function to extract numbers
def extract_numbers(image_path, machine_names):
    image = cv2.imread(image_path)
    results = model(image)

    detected_numbers = []  # Store (x, y, extracted_number)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            roi = image[y1:y2, x1:x2]  # Crop detected number region
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            text = pytesseract.image_to_string(gray, config='--psm 6')  # OCR Extraction

            detected_numbers.append((x1, y1, text.strip()))  # Store (X, Y, extracted number)

    # **Step 2: Sort detected numbers by Y first (top to bottom), then X (left to right)**
    detected_numbers.sort(key=lambda num: (num[1], num[0]))  

    extracted_data = []  # Store (Machine Name, Extracted Number) pairs

    # **Step 3: Assign machine names based on the sorted order**
    for i, (x, y, num) in enumerate(detected_numbers):
        machine_name = machine_names[i] if i < len(machine_names) else f"Unknown Machine {i+1}"
        extracted_data.append({"Machine": machine_name, "Extracted Number": num})

    return extracted_data

# Function to save extracted numbers with machine names to CSV
def save_to_csv(extracted_data):
    df = pd.DataFrame(extracted_data)  # Convert list of dicts to DataFrame
    df.to_csv(CSV_FILE_PATH, index=False)

# Function to send email with CSV file
def send_email(receiver_email):
    sender_email = "anushritambe07@gmail.com"  # Replace with your email
    sender_password = "dbdy nwiq vpxb danv"  # Replace with your email password

    msg = EmailMessage()
    msg['Subject'] = "Extracted Numbers CSV"
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content("Attached is the extracted numbers CSV file.")

    with open(CSV_FILE_PATH, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="csv", filename="extracted_numbers.csv")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)

# Streamlit Web App
def main():
    st.title("🔢 Manufacturing Count Extraction")
    st.write("📤 Upload an image to extract numbers automatically.")
    
    # Let user select board type
    board_type = st.radio("Choose Board", ["5-Machine Board", "21-Machine Board"])

    # Determine machine names based on selection
    machine_names = machine_names_5 if board_type == "5-Machine Board" else machine_names_21

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(file_path, caption="Uploaded Image", use_column_width=True)
        st.write("🔄 Processing...")

        extracted_data = extract_numbers(file_path, machine_names)
        save_to_csv(extracted_data)

        st.success("✅ Numbers Extracted Successfully!")
        st.write("### Extracted Numbers:")
        st.write(extracted_data)

        st.download_button(label="⬇ Download CSV", data=open(CSV_FILE_PATH, "rb"), file_name="extracted_numbers.csv", mime="text/csv")

        email = st.text_input("📧 Enter email to send CSV:")
        if st.button("📤 Send CSV via Email"):
            send_email(email)
            st.success(f"📩 Email sent to {email} successfully!")

if __name__ == "__main__":
    main()


# Function to send email with CSV file
def send_email(receiver_email):
    sender_email = "anushritambe07@gmail.com"  # Replace with your email
    sender_password = "dbdy nwiq vpxb danv"  # Replace with your email password

    msg = EmailMessage()
    msg['Subject'] = "Extracted Numbers CSV"
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content("Attached is the extracted numbers CSV file.")

    with open(CSV_FILE_PATH, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="csv", filename="extracted_numbers.csv")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, sender_password)
        server.send_message(msg)