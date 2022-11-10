import cv2, os, torch
import pandas as pd
from glob import glob

# A function that receives a path to an image
# It returns a tuple (regular_image, gray_image)
def get_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (image, gray)

# A function that receives a path to a file with annotations and
# returns an array of last 4 (x-center, y-center, width, height)
def get_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            annotations = line.split()[-4:]
    return [float(x) for x in annotations]

# A function that receives an image and coordinates of ears and
# draws a rectangle around them and displays the image
def draw_ears(image, ears, annotations):
    frame = []
    # Draw the rectangle we get from the predictor
    prediction_x = 0
    prediction_y = 0
    for (x, y, w, h) in ears:
        start = (x,y)
        end = (x+w, y+h)
        prediction_x = (int(x+w/2))
        prediction_y = int(y+h/2)
        print(type(image))
        frame = cv2.rectangle(image, start, end, (0,0,255), 2)
    
    # Draw the rectangle we get from the text file
    # First turn the annotations into a useful value
    h, w, _ = image.shape
    width = int(annotations[2] * image.shape[1])
    height = int(annotations[3] * image.shape[0])

    center_x = int(annotations[0]*w)
    center_y = int(annotations[1]*h)
    top_left = (int(center_x - width//2), int(center_y - height//2))
    bot_right = (int(center_x + width//2), int(center_y + height//2))

    frame = cv2.rectangle(frame, top_left, bot_right, (0, 255, 0), 2)

    # Compare the centers of the predictions and the annotations
    #print(f"Prediction center: ({prediction_x},{prediction_y})")
    #print(f"Annotation center: ({center_x},{center_y})")
    
    cv2.imshow('Ears', frame)
    cv2.waitKey(0)

# A function that receives the ground truth, predicted rectangle and the image and
# returns the IoU of that prediction
def get_iou(ground_truth, prediction):
    # Get the coordinates of the intersection 
    x1 = max(ground_truth[0], prediction[0])
    y1 = max(ground_truth[1], prediction[1])
    x2 = min(ground_truth[2], prediction[2])
    y2 = min(ground_truth[3], prediction[3])

    # Compute area of intersection
    intersection_area = max(0, x2-x1 + 1) * max(0, y2-y1 + 1)

    # Compute the area of the 2 other rectangles
    truth_area = (ground_truth[2] - ground_truth[0] + 1) * (ground_truth[3] - ground_truth[1] +1)
    prediction_area = (prediction[2] - prediction[0] + 1) * (prediction[3] - prediction[1] +1)

    #print(f"Truth area: {truth_area}")
    #print(f"Prediction area: {prediction_area}")
    #print(f"Intersection area: {intersection_area}")
    
    # Compute and return the IoU
    return intersection_area / (truth_area + prediction_area - intersection_area)

# A function that receives the text file coordinates, coordinates from the predictions and the image.
# Returns 2 arrays, each containing the x1, y1, x2, y2 coordinates (in pixels) respectively
def get_rectangle_coordinates(annotations, ears, image_shape, image):
    prediction = [0,0,0,0]
    ground_truth = [0,0,0,0]

    for (x, y, w, h) in ears:
        prediction[0] = x
        prediction[1] = y
        prediction[2] = x + w
        prediction[3] = y + h

    h, w = image_shape
    width = int(annotations[2] * w)
    height = int(annotations[3] * h)

    center_x = int(annotations[0]*w)
    center_y = int(annotations[1]*h)
    top_left = (int(center_x - width//2), int(center_y - height//2))
    bot_right = (int(center_x + width//2), int(center_y + height//2))

    ground_truth[0] = top_left[0]
    ground_truth[1] = top_left[1]
    ground_truth[2] = bot_right[0]
    ground_truth[3] = bot_right[1]

    # Comment these 4 lines
    frame = cv2.rectangle(image, (prediction[0], prediction[1]), (prediction[2], prediction[3]), (0,0,255), 2)
    frame = cv2.rectangle(frame, (ground_truth[0], ground_truth[1]), (ground_truth[2], ground_truth[3]), (0,255,0), 2)
    cv2.imshow("Predictions", frame)
    cv2.waitKey(0)

    return ground_truth, prediction


# A function that receives a path to a directory and 
# returns all the .pngs in that folder
def get_images(folder_path):
    return glob(f"{path}/*.png")


# A function that receives the path to an image, 
# makes the haar cascade ear prediction and compares the results to the ground truth.
# It returns the IoU
def compute_for_image_haar(image_path, detector_left, detector_right):
    annotations_path = image_path[:-4] + ".txt"
    #print(annotations_path)
    img, gray = get_image(image_path)
    annotations = get_annotations(annotations_path)

    # First try to detect left ears
    ears = detector_left.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5,
    		minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # If left ears are not found, try right ears
    if(len(ears) == 0):
        ears = detector_right.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5,
    		minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    iou = 0
    # If we get a result, compute the IoU and return it
    if(len(ears) > 0):
        image_shape = (len(img), len(img[0]))
        ground_truth, prediction = get_rectangle_coordinates(annotations, ears, image_shape, img)
        iou = get_iou(ground_truth, prediction)
        return iou
    # If we didn't find an ear, return None so this result can be ignored
    return None

# A function that receives the path to an image and a yolo model, 
# makes the yolo ear prediction and compares the results to the ground truth.
# It returns the IoU
def compute_for_image_yolo(image_path, model):
    result = model(image_path)
    tensor = result.xywh[0]
    # If the model didn't find an ear, return None so this result can be ignored
    if len(tensor) == 0:
        return None
    # Otherwise, convert the values we got from the model into an array if integers
    ears = restructure_yolo_data(tensor)
    image = cv2.imread(image_path)
    image_shape = (len(image), len(image[0]))

    annotations_path = image_path[:-4] + ".txt"
    annotations = get_annotations(annotations_path)
    ground_truth, prediction = get_rectangle_coordinates(annotations, ears, image_shape, image)
    iou = get_iou(ground_truth, prediction)
    print(iou)


# A function that receives yolo output and returns and array containing
# one array of 2 points
def restructure_yolo_data(yolo_tensor):
    array = yolo_tensor.numpy()[0]
    array[0] = array[0] - array[2]/2
    array[1] = array[1] - array[3]/2
    array = [int(x) for x in array][:4]
    return [array]

detector_left = cv2.CascadeClassifier("files\\haarcascade_mcs_leftear.xml")
detector_right = cv2.CascadeClassifier("files\\haarcascade_mcs_rightear.xml")
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path="files\\yolo5s.pt", force_reload=True)
path = "files\\test"
images = get_images(path)
iou_sum = 0
detected = 0

print(compute_for_image_haar("files\\test\\0501.png", detector_left, detector_right))
#compute_for_image_yolo("files\\test\\0501.png", yolo_model)
# Iterate through all the images
#for index, image in enumerate(images):
#    print(f"Image {index}/{len(images)}")
#    iou = compute_for_image_haar(image, detector_left, detector_right)
#    # If we get None, don't count this case
#    if(iou is not None):
#        iou_sum += iou
#        detected += 1


