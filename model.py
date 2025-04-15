import cv2
import numpy as np
from operator import itemgetter
from typing import Tuple, List
import joblib
import pandas as pd
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

static_model3 = models.resnet18(weights=None) 
static_model3.fc = nn.Linear(static_model3.fc.in_features, 3)  
state_dict = torch.load("new-only-pics3.pth", map_location=torch.device('cpu'), weights_only=True)
static_model3.load_state_dict(state_dict)
static_model3.eval()

static_model2 = joblib.load("best_logistic_regression_model2.pkl")
static_model5 = joblib.load("best_logistic_regression_model5.pkl")
static_model6 = joblib.load("best_ridge_model.pkl")

class ImageProcessor:
    """Class to handle image processing operations for object detection and measurement"""

    @staticmethod
    def safe_division(numerator: float, denominator: float) -> float:
        """Safely divides two numbers, returning 0 if denominator is zero"""
        return numerator / denominator if denominator != 0 else 0

    @staticmethod
    def hsv_background_subtraction(image: np.ndarray, 
                                 lower_hsv: Tuple[int, int, int], 
                                 upper_hsv: Tuple[int, int, int]) -> Tuple[np.ndarray, ...]:
        """Perform HSV-based background subtraction"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_inv = cv2.bitwise_not(mask)
        
        foreground = cv2.bitwise_and(image, image, mask=mask_inv)
        background = np.full_like(image, 255)
        background = cv2.bitwise_and(background, background, mask=mask)
        result = cv2.add(foreground, background)
        
        return hsv, mask, mask_inv, foreground, background, result

    @staticmethod
    def simple_thresholding(image: np.ndarray, 
                          threshold_value: int, 
                          threshold_type: int = cv2.THRESH_BINARY_INV) -> np.ndarray:
        """Apply simple thresholding to an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, threshold_type)
        return binary

    @staticmethod
    def preprocess_blob(image: np.ndarray, 
                       kernel_size: int, 
                       do_open: bool = False, 
                       do_close: bool = True, 
                       split_contour: bool = False) -> np.ndarray:
        """Preprocess binary image with morphological operations"""
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if do_open:
            image = cv2.erode(image, kernel)
            image = cv2.dilate(image, kernel)
        if do_close:
            image = cv2.dilate(image, kernel)
            image = cv2.erode(image, kernel)
        if split_contour:
            image = ImageProcessor.watershed_split_contours(image)
        
        return image

    @staticmethod
    def watershed_split_contours(binary_image: np.ndarray) -> np.ndarray:
        """Separate touching objects using watershed algorithm"""
        kernel = np.ones((3, 3), np.uint8)
        
        sure_bg = cv2.dilate(binary_image, kernel, iterations=2)
        dist_transform = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
        
        sure_fg = np.uint8(sure_fg)
        sure_bg = np.uint8(sure_bg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        rgb_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        segmented = cv2.watershed(rgb_image, markers)
        result = binary_image.copy()
        result[segmented == -1] = 0
        
        return result

    @staticmethod
    def min_gap_pixel(cropped_gray_image: np.ndarray, 
                     row_size: int) -> Tuple[float, List]:
        """Calculate minimum gap between vertical lines"""
        vertical_line_count = np.sum(cropped_gray_image == 0, axis=0)
        top_indices = np.argsort(vertical_line_count)[::-1]
        
        top_lines = [top_indices[0]]
        vertical_lines = [[(top_indices[0], 0), (top_indices[0], cropped_gray_image.shape[0])]]
        
        for idx in top_indices[1:]:
            if all(abs(existing - idx) >= 30 for existing in top_lines):
                vertical_lines.append([(idx, 0), (idx, cropped_gray_image.shape[0])])
                top_lines.append(idx)
                if len(top_lines) == 4:
                    break
        
        top_lines.sort()
        gaps = [abs(top_lines[i+1] - top_lines[i]) for i in range(len(top_lines)-1)]
        min_gap_range = [int((row_size/4032)*120), int((row_size/4032)*160)]
        valid_gaps = [gap for gap in gaps if min_gap_range[0] <= gap <= min_gap_range[1]]
        
        return sum(valid_gaps)/len(valid_gaps) if valid_gaps else 0, vertical_lines

def model1(image: np.ndarray) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """Main function to process image and extract measurements"""
    ROW, COL = image.shape[:2]
    
    # Constants
    LOWER_GREEN = np.array([35, 55, 55])
    UPPER_GREEN = np.array([85, 255, 255])
    GRAY_TH = 250
    
    # Initialize output images
    image1 = image.copy()  # For bounding rectangles
    image2 = image.copy()  # For inner/outer contours
    
    # Initial processing
    processor = ImageProcessor()
    _, _, _, _, _, subtract_bg = processor.hsv_background_subtraction(image, LOWER_GREEN, UPPER_GREEN)
    thresh = processor.simple_thresholding(subtract_bg, GRAY_TH)
    processed = processor.preprocess_blob(cv2.bitwise_not(thresh), kernel_size=10)
    
    # Second stage processing
    mask_inv = cv2.bitwise_not(processed)
    foreground = cv2.bitwise_and(image, image, mask=mask_inv)
    background = cv2.bitwise_and(np.full_like(image, 255), np.full_like(image, 255), mask=processed)
    final_image = cv2.add(foreground, background)
    
    gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Contour analysis
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    measurements = {
        'percentages': [],
        'blob_areas': [],
        'inner_areas': [],
        'diameters': [],
        'lengths': []
    }
    
    top_left = None
    roi1, roi2 = None, None
    for contour in contours:
        blob_area = cv2.contourArea(contour)
        min_area = int((COL / 4032) * 100000)
        max_area = int((COL / 4032) * 1500000)
        
        if min_area < blob_area < max_area:
            # Rectangle measurements and drawing
            rect = cv2.minAreaRect(contour)
            box = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(image1, [box], 0, (0, 0, 255), 10)  # Red rectangles on image1
            x, y, w, h = cv2.boundingRect(contour)
            if roi1 is None:
                roi1 = final_image[y:y+h, x:x+w]
            elif roi2 is None: 
                roi2 = final_image[y:y+h, x:x+w]
            
            width, height = rect[1]
            diameter, length = min(width, height), max(width, height)
            
            # Update top-left coordinate
            box_min = np.min(box, axis=0)
            if top_left is None or box_min[0] < top_left[0] or box_min[1] < top_left[1]:
                top_left = box_min
            
            # Gap calculation
            crop_start = (int((ROW/4032)*250), int((COL/3024)*200))
            crop_size = (int((ROW/4032)*500), int((COL/3024)*500))
            crop = image[crop_start[0]:min(crop_start[0] + crop_size[0], top_left[1]),
                        crop_start[1]:crop_start[1] + crop_size[1]]
            gray_crop = processor.simple_thresholding(crop, 80, cv2.THRESH_BINARY)
            min_gap, _ = processor.min_gap_pixel(gray_crop, ROW)
            ratio = 5/min_gap if min_gap != 0 else 0
            
            # Store measurements
            measurements['blob_areas'].append(blob_area)
            measurements['diameters'].append(diameter)
            measurements['lengths'].append(length)
            
            # Inner contour analysis and drawing
            x, y, w, h = cv2.boundingRect(contour)
            roi = thresholded[y:y+h, x:x+w]
            padded = cv2.copyMakeBorder(roi, 
                                      (ROW-h)//2, (ROW-h+1)//2, 
                                      (COL-w)//2, (COL-w+1)//2,
                                      cv2.BORDER_CONSTANT, 
                                      value=255)
            eroded = cv2.erode(padded, np.ones((5, 5), np.uint8))
            
            inner_contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            inner_areas = sorted([cv2.contourArea(c) for c in inner_contours], reverse=True) + [0, 0]
            measurements['inner_areas'].append(inner_areas[2])
            measurements['percentages'].append(processor.safe_division(inner_areas[2], inner_areas[1]) * 100)
            
            # Draw inner and outer contours on image2
            for contour2 in inner_contours:
                blob_area_in_nut = cv2.contourArea(contour2)
                cnt_translated = contour2 + np.array([[x - (COL-w)//2, y - (ROW-h)//2]])
                if blob_area_in_nut == inner_areas[1]:
                    cv2.drawContours(image2, [cnt_translated], 0, (0, 0, 255), 10)  # Red outer
                elif blob_area_in_nut == inner_areas[2]:
                    cv2.drawContours(image2, [cnt_translated], 0, (255, 0, 0), 10)  # Blue inner
    
    # Calculate averages
    ratio = ratio if 'ratio' in locals() else 1
    avg = lambda x: processor.safe_division(sum(x), max(len(x), 1))
    
    return (
        avg(measurements['diameters']) * ratio,
        avg(measurements['lengths']) * ratio,
        avg(measurements['blob_areas']) * ratio ** 2,
        avg(measurements['inner_areas']) * ratio ** 2,
        avg(measurements['percentages']),
        image1[top_left[1]+int(avg(measurements['lengths'])/2)-1000:top_left[1]+int(avg(measurements['lengths'])/2)+1000],
        image2[top_left[1]+int(avg(measurements['lengths'])/2)-1000:top_left[1]+int(avg(measurements['lengths'])/2)+1000],
        roi1, roi2
    )

def model2(actual_kernel_weight: float, pred_total_area: float, percentage: int) -> int:
    X_test_df = pd.DataFrame([[actual_kernel_weight,pred_total_area,percentage]], columns=['actual kernel weight (g)' ,'predicted average total area (mm)', 'Percentage'])
    y_pred = static_model2.predict(X_test_df)
    return y_pred[0]
    
def model3_predict(image, model):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_batch = input_batch.to(device)
    
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class
    
def model3(roi1: np.ndarray, roi2: np.ndarray) -> int:
    model3_1_pred_class = model3_predict(roi1[:,:,::-1], static_model3)
    model3_2_pred_class = model3_predict(roi2[:,:,::-1], static_model3)
    class_labels = [0, 50, 100] 
    return class_labels[model3_1_pred_class], class_labels[model3_2_pred_class]

def model5(pred_total_area: float, actual_total_weight: float) -> int:
    X_test_df = pd.DataFrame([[pred_total_area,actual_total_weight]], columns=['predicted average total area (mm)', 'actual total weight (g)'])
    y_pred = static_model5.predict(X_test_df)
    return y_pred[0]

def model6(pred_total_area: float) -> int:
    X_test_df = pd.DataFrame([[pred_total_area]], columns=['predicted average total area (mm)'])
    y_pred = static_model6.predict(X_test_df)
    return y_pred[0]

def weight_voting(model2: int, model3_1: int, model3_2: int, model5: int) -> int:
    model_weight = {"model2": 0.500, "model3_1": 1.138, "model3_2": 0.864, "model5": 0.499}
    count = {50: 0.0, 100: 0.0, 0: 0.0}
    count[model2] += model_weight["model2"]
    count[model3_1] += model_weight["model3_1"]
    count[model3_2] += model_weight["model3_2"]
    count[model5] += model_weight["model5"]
    predicted_label = max(count, key=count.get)
    return predicted_label

def hello_world2():
    return "2"
    
def hello_world():
    return "hello world"+hello_world2()
