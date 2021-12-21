import cv2
import numpy as np

def get_image(image_path):
    image = cv2.imread(image_path)
    return image

def get_splitline_coordinates(dilated_image, conv_width=3, trashold_value=250, y_start=21):
    for i in range(y_start, dilated_image.shape[0] - conv_width):
        conv_average = dilated_image[i:i + conv_width].mean()
        if conv_average > trashold_value:
            return i + conv_width

def get_processed_image(image,morph_kernel_shape):
    edgy_image = cv2.Canny(image,240,255,L2gradient=False)
    kernel = np.ones(morph_kernel_shape, np.uint8)
    dilated_image = cv2.dilate(edgy_image, kernel, iterations=1)
    return dilated_image
        
def get_header_body_images(image, split_line):
    header = image[:split_line]
    body = image[split_line:]
    return header, body

def split_table(image_path,morph_kernel_shape=(5,5)):
    image = get_image(image_path)
    
    dilated_image = get_processed_image(image,morph_kernel_shape)
    
    split_line = get_splitline_coordinates(dilated_image)
    header, body = get_header_body_images(image, split_line)

    return header, body