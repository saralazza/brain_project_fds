import numpy as np
import cv2
import os
import torch

def find_rectangle(image, gray_image):
    # Find edges and enanche them
    edges = cv2.Canny(gray_image, 50, 400)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours = np.vstack(contours)

    # Find the rectangle that contains all contours
    x, y, w, h = cv2.boundingRect(all_contours)
    
    return x, y, w, h

def center_image_in_square(image, square_size):  
    # Check if the image is smaller than or equal to the square
    if image.shape[0] > square_size or image.shape[1] > square_size:        
        raise ValueError("L'immagine è più grande della matrice quadrata specificata!")
    # Create the square matrix of the pixel with the min value
    square_matrix = np.full((square_size, square_size), np.min(image))

    # Calculate offsets to center the image   
    offset_y = (square_size - image.shape[0]) // 2
    offset_x = (square_size - image.shape[1]) // 2

    # Copy the image in the center  
    square_matrix[offset_y:offset_y + image.shape[0], offset_x:offset_x + image.shape[1]] = image
    return square_matrix

def process_images(paths, output_folder):
    for img_path in paths:
        # Laod image and the gray scale image
        image = cv2.imread(img_path)
        gray_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Find the rectangle
        x, y, w, h = find_rectangle(image, gray_image)

        # Cut the rectangle image
        gray_image = gray_image[y:y+h,x:x+w]

        # Center the rectangle image in a squared black image
        d = max(h,w)
        square_size = d + 50
        centered_image = center_image_in_square(gray_image, square_size)

        # Resize the squared image
        resized_img = cv2.resize(centered_image, (128, 128))

        # Save the squared image
        ext = img_path.split('\\')[-1]
        path = output_folder + ext
        cv2.imwrite(path, resized_img)

def load_preprocessed_image(path):
    # Load preprocessed images
    image_paths_yes_prepocessed = []
    image_paths_no_prepocessed = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            image_path = os.path.join(dirname, filename)
            if dirname.lower().endswith("no"):
                image_paths_no_prepocessed.append(image_path)
            else:
                image_paths_yes_prepocessed.append(image_path)
    return image_paths_yes_prepocessed, image_paths_no_prepocessed