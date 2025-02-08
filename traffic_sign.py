import cv2
import numpy as np
from tensorflow.keras.models import load_model
import torch
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Traffic sign classes
classes = {
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)', 
    2:'Speed limit (50km/h)', 
    3:'Speed limit (60km/h)', 
    4:'Speed limit (70km/h)', 
    5:'Speed limit (80km/h)', 
    6:'End of speed limit (80km/h)', 
    7:'Speed limit (100km/h)', 
    8:'Speed limit (120km/h)', 
    9:'No passing', 
    10:'No passing veh over 3.5 tons', 
    11:'Right-of-way at intersection', 
    12:'Priority road', 
    13:'Yield', 
    14:'Stop', 
    15:'No vehicles', 
    16:'Veh > 3.5 tons prohibited', 
    17:'No entry', 
    18:'General caution', 
    19:'Dangerous curve left', 
    20:'Dangerous curve right', 
    21:'Double curve', 
    22:'Bumpy road', 
    23:'Slippery road', 
    24:'Road narrows on the right', 
    25:'Road work', 
    26:'Traffic signals', 
    27:'Pedestrians', 
    28:'Children crossing', 
    29:'Bicycles crossing', 
    30:'Beware of ice/snow',
    31:'Wild animals crossing', 
    32:'End speed + passing limits', 
    33:'Turn right ahead', 
    34:'Turn left ahead', 
    35:'Ahead only', 
    36:'Go straight or right', 
    37:'Go straight or left', 
    38:'Keep right', 
    39:'Keep left', 
    40:'Roundabout mandatory', 
    41:'End of no passing', 
    42:'End no passing veh > 3.5 tons'
}

# Load YOLO model
model_yolo = YOLO('E:/project/Computer vision sem5/best.pt')

# Load CNN model
model_cnn = load_model('model1.h5')


def enhance_image(img, alpha=1.3, beta=30, brightness_factor=1.2):
    # 1. Convert to HSV for brightness adjustment
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    brightened_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 2. Adjust contrast and brightness using scaling
    enhanced_img = cv2.convertScaleAbs(brightened_img, alpha=alpha, beta=beta)
    
    # 3. Apply sharpening filter to reduce blur
    sharpening_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    sharpened_img = cv2.filter2D(enhanced_img, -1, sharpening_kernel)

    return sharpened_img

    
def preprocess_for_cnn(img):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
    # Resize to 30x30
    img_resized = cv2.resize(img, (32, 32))
    img_normalized = img_resized.astype('float32') / 255.0
    img_preprocessed = np.expand_dims(img_normalized, axis=0)
    
    return img_preprocessed

def process_image(image_path):
    # Read image
    img = cv2.imread(image_path)
    original = img.copy()
    
    results = model_yolo(img)
    result = results[0]
    
    # Check if any detections
    if len(result.boxes) == 0:
        print("No traffic signs detected")
        return 0,0,None,0,0,None,0,0
    
    # Get the first detected sign
    box = result.boxes[0]  # Get first detection
    x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    sign = original[y1:y2, x1:x2]
    
    # Ensure sign was properly extracted
    if sign.size == 0:
        print("Error: Could not extract sign from image")
        return
        
    # Just image 
    img_arr = preprocess_for_cnn(img)  
    s_prediction = model_cnn.predict(img_arr, verbose = 0)
    s_class = np.argmax(s_prediction)
    s_meaning = classes[s_class]

    # without preprocessing of sign  input to cnn 
    i_preprocess = preprocess_for_cnn(sign)
    i_predict = model_cnn.predict(i_preprocess,verbose = 0)
    i_class = np.argmax(i_predict)
    i_meaning  = classes[i_class]

    # Enhance the extracted sign
    enhanced_sign = enhance_image(sign)
    preprocessed = preprocess_for_cnn(enhanced_sign)
    prediction = model_cnn.predict(preprocessed, verbose=0)  # Added verbose=0 to reduce output
    predicted_class = np.argmax(prediction)
    sign_meaning = classes[predicted_class]

    return s_class,s_meaning,sign,i_class,i_meaning,enhanced_sign,predicted_class,sign_meaning



    
    
  


