import cv2
import sys
import mediapipe as mp
import base64
import argparse
import warnings
import logging
import os
import json
from openai import OpenAI  # Ensure you have this import

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all logs below ERROR level

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set logging level to WARNING to suppress INFO and DEBUG messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)


client = OpenAI(api_key='')

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    success, frame = cap.read()
    count = 0
    
    validations = {
        "validations": []
    }

    with open('./validation_file.json', 'r') as json_file:
        validation_data = json.load(json_file)

    created_images = []  # List to keep track of created images

    while success:
        if count % int(frame_rate) == 0:  # Extract 1 frame per second
            frame_filename = f"frame_{count}.jpg"
            cv2.imwrite(frame_filename, frame)
            created_images.append(frame_filename)  # Track created image
            print(f"Extracted: {frame_filename}")  # Print extraction step
            
            # Process the extracted frame
            description = generate_frame_description(frame, count)
            glasses_check_result = check_glasses_with_openai(frame_filename)

                        # Create a validations object to store results

            # Clean the glasses_check_result to ensure it's a boolean
            cleaned_glasses_result = glasses_check_result.strip().lower() == "true"
            validations["validations"].append({
                "mouth_open": description,
                "glasses": cleaned_glasses_result
            })
            print(f"Check Result for {frame_filename}: {{'mouth_open': {description}, 'glasses': {cleaned_glasses_result}}}")
            
        success, frame = cap.read()
        count += 1
    
    # Compare validations with the loaded JSON data
    if validations == validation_data:
        print("\033[92mValidations match the JSON data.\033[0m")  # Green
    else:
        print("\033[91mValidations do not match the JSON data.\033[0m")  # Red

    # Remove created images
    for image in created_images:
        os.remove(image)
    
    cap.release()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def detect_mouth_open(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Calculate mouth opening based on landmark distances
            upper_lip = landmarks.landmark[13]  # Top of upper lip
            lower_lip = landmarks.landmark[14]  # Bottom of lower lip
            mouth_open_distance = abs(upper_lip.y - lower_lip.y)

            # Define a threshold for "open mouth" (tune this value)
            if mouth_open_distance > 0.05:
                return True
    return False

def generate_frame_description(image, frame_num):
    is_mouth_open = detect_mouth_open(image)

    return is_mouth_open

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def check_glasses_with_openai(image_path):
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is there a person wearing glasses in this image? just answer true or false."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    
    return response.choices[0].message.content  # Return only the content

if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else '../briefs_video.MP4'
    extract_frames(video_path)
