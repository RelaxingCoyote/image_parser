import re
import json
import os
import cv2

PATTERN_FORMULA_DESC = re.compile("\d{1,3}\.{0,1}\d{0,3}[a-z]{0,1}|/^[a-z]{1}$/"
                                                "|/^[I,V,X,L]{1,4}[a-z]{0,1}$/",re.IGNORECASE)

FORMULA_FRAME_PADDING = {
    'y_pad' : 0.05,
    'x_pad' : 0.01
}

PAGE_PATTERN = '_{page_number}.png'

def create_required_dirs_if_dont_exist(dir_path):
    if not os.path.exists(dir_path):
      os.makedirs(dir_path)

def find_page_image_by_number(pages_list: list, page_number: int) -> str:
    page_pattern = PAGE_PATTERN.format(page_number=page_number)
    for page_name in pages_list:
        if page_pattern in page_name:
            return page_name

    raise ValueError('There are no pages with the pattern {page_pattern}'.format())

def get_required_page(path_out_temp, pages_list, page_number):
    page_with_the_formula = find_page_image_by_number(pages_list, page_number)
    page_path = os.path.join(path_out_temp, page_with_the_formula)

    return cv2.imread(page_path)

def get_scale_coeficients(page_image,pdf_page_width,pdf_page_height):
    x_scale_coefficient = page_image.shape[1] / pdf_page_width
    y_scale_coefficient = page_image.shape[0] / pdf_page_height

    return x_scale_coefficient, y_scale_coefficient

def get_formula_coords_padding(y_1,y_2,x_1,x_2,y_pad,x_pad):
    return y_1 * (1 - y_pad/2), y_2 * (1 + y_pad / 2), x_1 *(1 - x_pad / 2), x_2 *(1 + x_pad / 2)

def get_scaled_coordinates_for_cv2(x_scale_coefficient, y_scale_coefficient,x_1,y_1,x_l,y_l):
    x_1 = x_1 * x_scale_coefficient
    y_1 = y_1 * y_scale_coefficient
    x_2 = x_1 + x_l * x_scale_coefficient
    y_2 = y_1 + y_l * y_scale_coefficient

    # y_1,y_2,x_1,x_2 = get_formula_coords_padding(y_1, y_2, x_1, x_2,**FORMULA_FRAME_PADDING)

    return int(y_1), int(y_2), int(x_1), int(x_2)

def get_formula_image(path_out_temp, pages_list, pdf_page_width, pdf_page_height,page_number, x_1,y_1,x_l,y_l):
    page_image = get_required_page(path_out_temp,pages_list,page_number)
    x_scale_coefficient, y_scale_coefficient = get_scale_coeficients(page_image,pdf_page_width,pdf_page_height)
    y_1,y_2,x_1,x_2 = get_scaled_coordinates_for_cv2(x_scale_coefficient,y_scale_coefficient,x_1,y_1,x_l,y_l)

    return page_image[int(y_1):int(y_2),int(x_1):int(x_2)]

def write_to_json(file_directory, item_name, item_description="", object_type='figure'):
    object_id = f'{object_type}_id'
    item = {object_id:item_name,'describe':item_description}
    field_name = f"{object_type}s"
    if not os.path.exists(os.path.join(file_directory,"temp.json")):
        fig_dict = {field_name:[item]}
        with open(os.path.join(file_directory,"temp.json"),'w') as write_file:
            json.dump(fig_dict, write_file)
    else:
        with open(os.path.join(file_directory,"temp.json"), "r") as write_file:
            fig_dict = json.load(write_file)
            is_absent = True
            for count in range(len(fig_dict[field_name])):
                if item_name == fig_dict[field_name][count][object_id]:
                    fig_dict[field_name][count] = item
                    is_absent = False

            if is_absent == True:
                fig_dict[field_name].append(item)

        with open(os.path.join(file_directory,"temp.json"), "w") as write_file:
            json.dump(fig_dict, write_file)

def save_untitled_formula(formula_dir, formula_image):
    untitled_formulas_dir = os.path.join(formula_dir,"untitled_formulas")
    create_required_dirs_if_dont_exist(untitled_formulas_dir)
        
    # ???????????????? ???????????? ????????????????????????
    list_images = os.listdir(untitled_formulas_dir)
    # ?????????????????? ???????????? ?????????? ???????????????? ?????????????????? ??????????????
    list_images.sort()
        
    # ???????????? ????????????????
    image_name_pattern = "untitled_"
    # ?????????????????? ?????????????? ?????????????????????????????? ??????????????
    list_images = [el for el in list_images if image_name_pattern in el]

    if len(list_images)==0:
        # ???????????? ?????????????????????? ?????? ????????????????
        image_name = "untitled_0"
    else:
        # ???????????????? ?????? ???????????????????? ?????????????????????? ?????? ????????????????????
        image_name = list_images[-1][:-4]
        # ???????????????? ?????????? ??????????????????????
        image_number = image_name.replace(image_name_pattern,"")
        image_number = int(image_number) + 1
        # ?????? n-???? ??????????????????????
        image_name = image_name_pattern + str(image_number)

    cv2.imwrite(f"{untitled_formulas_dir}/{image_name}.png", formula_image)

def save_formulas(path_out):
    with open(f"{path_out}/temp.json") as read_file:
        formulas_dict = json.load(read_file)

    pdf_page_width = float(formulas_dict["lrx"])
    pdf_page_height = float(formulas_dict["lry"])
    path_temp = os.path.join(path_out,'temp')

    pages = os.listdir(path_temp)
    pages.sort()

    save_formula_directory = os.path.join(path_out, 'formulas')

    for key,value in formulas_dict['coordinates'].items():
        formula_coordinates = [float(item) if item != value.split(',')[0] else int(item) for item in value.split(',')]
        formula_image = get_formula_image(path_temp,pages,pdf_page_width,pdf_page_height,*formula_coordinates)
        try:
            search_result = re.search(PATTERN_FORMULA_DESC,key)
            formula_num = search_result.group(0)
            formula_id = f"formula_{formula_num}"
            create_required_dirs_if_dont_exist(save_formula_directory)
            cv2.imwrite(f"{save_formula_directory}/{formula_id}.png", formula_image)
            write_to_json(save_formula_directory,formula_id,object_type="formula")
        except AttributeError:
            save_untitled_formula(save_formula_directory,formula_image)