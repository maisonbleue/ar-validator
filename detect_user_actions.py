import cv2
import mediapipe as mp

import argparse
import base64
import requests

from openai import OpenAI  # Change this line

client = OpenAI(api_key='')

# Set your OpenAI API key

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

def detect_glasses(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glasses = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(glasses) > 0:
        return True
    return False

def generate_frame_description(image, frame_num):
    is_mouth_open = detect_mouth_open(image)
    has_glasses = detect_glasses(image)

    description = f"Frame {frame_num}: "
    description += "Mouth is open, " if is_mouth_open else "Mouth is closed, "
    description += "Glasses are present." if has_glasses else "No glasses."

    return description

# Function to encode the image in base64
def encode_image(image_path):
    """
    Encodes the image file at the given path into a base64 string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to check for glasses using OpenAI API
def check_glasses_with_openai(image_path):  # {{ edit_2 }}
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(  # {{ edit_3 }}
        model="gpt-4o",  # Use the appropriate model
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is there a person wearing glasses in this image? answer with just true or false."},
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
    )  # {{ edit_4 }}

    return response.choices[0].message.content  # {{ edit_5 }}

if __name__ == "__main__":  # {{ edit_2 }}
    parser = argparse.ArgumentParser(description='Detect user actions in an image.')
    parser.add_argument('image', type=str, help='Path to the input image')
    parser.add_argument('frame_num', type=int, help='Frame number for the description')
    args = parser.parse_args()

    image = cv2.imread(args.image)  # {{ edit_3 }}
    description = generate_frame_description(image, args.frame_num)  # {{ edit_4 }}

    glasses_check_result = check_glasses_with_openai(args.image)
    print(glasses_check_result)  # Output the result
    print(description)  # {{ edit_5 }}