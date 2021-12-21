import cv2
import numpy as np

SPLITLINE_REGION_WIDTH = 20
DISTANCE_BETWEEN_MULTYINDEX_LINES = 50
HEADER_SPLITLINE_LENGTH_TO_IMAGE_WIDTH_RATIO = 0.11

def get_processed_image(image,morph_kernel_shape):
    edgy_image = cv2.Canny(image,240,255,L2gradient=False)
    kernel = np.ones(morph_kernel_shape, np.uint8)
    dilated_image = cv2.dilate(edgy_image, kernel, iterations=1)
    return dilated_image

def get_table_header_lines(image):
    lines = cv2.HoughLinesP(image, 1, np.pi/180, 30, maxLineGap=1)
    return [line[0] for line in lines if get_line_length(*line[0])\
        > HEADER_SPLITLINE_LENGTH_TO_IMAGE_WIDTH_RATIO * image.shape[1]]

def is_upper_header_plank(x_1,y_1,x_2,y_2):
    return y_1 < 24

def is_header_body_splitline(image_h,x_1,y_1,x_2,y_2):
    return y_1 > image_h - SPLITLINE_REGION_WIDTH

def are_lines_the_same(x_1_a,y_1_a,x_2_a,y_2_a,x_1_b,y_1_b,x_2_b,y_2_b):
    return abs(x_1_a - x_1_b) <= DISTANCE_BETWEEN_MULTYINDEX_LINES

def get_line_length(x_1,y_1,x_2,y_2):
    return np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)

def is_in_list(filtered_list,x_1_l,y_1_l,x_2_l,y_2_l):
    for item in filtered_list:
        if are_lines_the_same(*item,x_1_l,y_1_l,x_2_l,y_2_l):
            return True
    return False

def filter_table_header_lines(table_header_lines_list,image_h):
    table_header_lines_list_filtered = []
    for i in range(len(table_header_lines_list)):
        if not is_upper_header_plank(*table_header_lines_list[i])\
            and not is_header_body_splitline(image_h,*table_header_lines_list[i]):
            if len(table_header_lines_list_filtered)== 0:
                table_header_lines_list_filtered.append(table_header_lines_list[i])
            elif not is_in_list(table_header_lines_list_filtered,*table_header_lines_list[i]):
                table_header_lines_list_filtered.append(table_header_lines_list[i])
                
    return table_header_lines_list_filtered

def get_filtered_table_header_lines(header_image, morph_kernel_shape=(5,5)):
    dilated_header = get_processed_image(header_image,morph_kernel_shape)
    table_header_lines_list = get_table_header_lines(dilated_header)
    filtered_split_lines_list = filter_table_header_lines(table_header_lines_list,header_image.shape[0])
    return filtered_split_lines_list