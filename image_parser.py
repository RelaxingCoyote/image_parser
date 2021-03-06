from abc import abstractmethod
import os
import re
import json
from tqdm import tqdm
import numpy as np
import cv2
import layoutparser as lp
from pdf2image import convert_from_path
# from modules.formula_extractor import save_formulas

from warnings import filterwarnings

filterwarnings('ignore')

SECOND_TO_FIRST_COLUMN_RATIO = 5.0

def sort_layout_blocks(layout_blocks):
    for i in range(len(layout_blocks)):
        for j in range(i,len(layout_blocks)):
            if not layout_blocks[j]['x_1'] / layout_blocks[i]['x_1'] > SECOND_TO_FIRST_COLUMN_RATIO:
                if layout_blocks[j]['y_1'] <= layout_blocks[i]['y_1'] :
                    layout_blocks[i], layout_blocks[j] = layout_blocks[j], layout_blocks[i]
                elif layout_blocks[j]['x_1'] / layout_blocks[i]['x_1'] < 1 / SECOND_TO_FIRST_COLUMN_RATIO:
                    layout_blocks[i], layout_blocks[j] = layout_blocks[j], layout_blocks[i]

    return layout_blocks

def get_coordinates(bounding_boxes_block):
    x_1,y_1 = int(np.floor(bounding_boxes_block['x_1'])),int(np.floor(bounding_boxes_block['y_1']))
    x_2,y_2 = int(np.ceil(bounding_boxes_block['x_2'])),int(np.ceil(bounding_boxes_block['y_2']))

    return y_1, y_2, x_1, x_2

def get_bigger_number(f_1,f_2):
    return f_1 if f_1 > f_2 else f_2

def get_smaler_number(f_1,f_2):
    return f_1 if f_1 < f_2 else f_2

def get_union_cooridnates(layout_block_1,layout_block_2,union_impossible_return_block='upper'):
    y_1_upper, y_2_upper, x_1_upper, x_2_upper = get_coordinates(layout_block_1)
    y_1_lower, y_2_lower, x_1_lower, x_2_lower = get_coordinates(layout_block_2)

    if not x_1_upper / x_1_lower < 1 / SECOND_TO_FIRST_COLUMN_RATIO or not x_1_upper / x_1_lower > SECOND_TO_FIRST_COLUMN_RATIO:
        y_1_union = get_smaler_number(y_1_upper,y_1_lower)
        x_1_union = get_smaler_number(x_1_upper,x_1_lower)

        y_2_union = get_bigger_number(y_2_upper,y_2_lower)
        x_2_union = get_bigger_number(x_2_upper,x_2_lower)

        return y_1_union, y_2_union,  x_1_union, x_2_union

    if union_impossible_return_block == 'upper':
        return y_1_upper, y_2_upper, x_1_upper, x_2_upper
    else:
        return y_1_lower, y_2_lower, x_1_lower, x_2_lower

def get_fig_info(pattern,text):
    result = re.search(pattern,text)
    return result.group(0)

class ImageParser():

    def __init__(self):
        self.model = lp.Detectron2LayoutModel(
                                                'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.80],
                                                label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
                                                )                                                    
        self.pattern_fig = re.compile("Fig. \d*|Figure \d*|Scheme \d*|Chart \d*",re.IGNORECASE)
        self.pattern_table = re.compile("Table \d*[^.,]",re.IGNORECASE)
        self.pattern_fig_desc = re.compile("Fig. \d*[\.\:] [A-Z][\s\S]+|"
                                            "Figure \d*[\.\:] [A-Z][\s\S]+|Scheme \d* [A-Z][\.\:][\s\S]+"
                                            "|Chart \d* [A-Z][\.\:][\s\S]+",re.IGNORECASE)
        self.pattern_table_desc = re.compile("Table \d*[\s\S]+?(?=\n\n)",re.IGNORECASE)

        self.ocr_agent = lp.TesseractAgent(languages='eng')

    def logger(self,what_happened,where_happened,path_out):
        logger_path = f"{path_out}/logs/log.txt"
        logger_folder = f"{path_out}/logs"
        if not os.path.exists(logger_folder):
            os.makedirs(logger_folder)
            f = open(logger_path ,"w")
        else:
            f = open(logger_path,"a")
        f.write(f"{what_happened} : {where_happened}\n")
    
    def write_to_json(self,file_directory,item_name,item_description="",object_type='figure'):
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

    def read_temp(self,file_directory,object_type="figures"):
        temp_path = os.path.join(file_directory,object_type,"temp.json")
        if os.path.exists(temp_path):
            with open(temp_path, "r") as read_file:
                fig_dict = json.load(read_file)
            # fig_dict = sorted()
            return fig_dict[object_type]
        else:
            return []
        
    
    def get_figures_with_info(self,layout_blocks_sorted,page_path,figures_path):
        blocks_volume = len(layout_blocks_sorted)
        for block_num in range(blocks_volume):
            if layout_blocks_sorted[block_num]['type'] == "Figure":
                y_1, y_2, x_1, x_2 = get_coordinates(layout_blocks_sorted[block_num])
                page_image = cv2.imread(page_path)

                page_width = page_image.shape[1]
                page_height = page_image.shape[0]
                figure_width = x_2-x_1
                fw_to_iw = figure_width/page_width
                fh_to_ph = (y_2-y_1)/page_height

                # ?????????????????????? ???? ?????? ????????????????
                if (fh_to_ph > 0.785) and (fw_to_iw > 0.60):
                    fig_union_image = cv2.rotate(page_image, cv2.cv2.ROTATE_90_CLOCKWISE)
                    # ???????????????????????????? ??????????????????????
                    figure = page_image[int(y_1):int(y_2),int(x_1):int(x_2)]
                    figure = cv2.rotate(figure, cv2.cv2.ROTATE_90_CLOCKWISE)
                # ?????????????????????? ???? ???? ?????? ????????????????
                else:
                    figure = page_image[int(y_1):int(y_2),int(x_1):int(x_2)]
                    if block_num < blocks_volume - 1:
                        y_1_union, y_2_union, x_1_union, x_2_union = get_union_cooridnates(layout_blocks_sorted[block_num],
                                                                                            layout_blocks_sorted[block_num+1])
                    else:
                        y_1_union, y_2_union, x_1_union, x_2_union = y_1, y_2, x_1, x_2
                    fig_union_image = page_image[y_1_union:y_2_union,x_1_union:x_2_union]
                try:
                    figure_text = self.ocr_agent.detect(fig_union_image)
                    result = re.search(self.pattern_fig,figure_text)
                    fig_num = result.group(0)
                    fig_name = fig_num.replace(" ","_").replace(".","").lower()
                    if not os.path.exists(f"{figures_path}/figures"):
                        os.makedirs(f"{figures_path}/figures")

                    cv2.imwrite(f"{figures_path}/figures/{fig_name}.png", figure)
                    self.write_to_json(f"{figures_path}/figures",fig_name,object_type ='figure')
                    description = re.search(self.pattern_fig_desc,figure_text)
                    description = description.group(0)
                    description = description.replace(fig_num,"")
                    description = description.strip("\n\f")
                    description = description.strip("\n\x0c")
                    description = description[2:]
                    self.write_to_json(f"{figures_path}/figures",fig_name,item_description=description,object_type ='figure')
                except AttributeError:
                    self.save_image_as_it_is(layout_blocks_sorted[block_num],page_path,
                                figures_path,object_type="figures")

    # ?????????????????? ?????????????????????? ?????? undefined_n
    def save_image_as_it_is(self,fig_block,image_path,
                            figures_path,object_type="figures"):
        # ?????????????? ???????????????????? ??????????????????????
        x_1,y_1 = np.floor(fig_block['x_1']),np.floor(fig_block['y_1'])
        x_2,y_2 = np.ceil(fig_block['x_2']),np.ceil(fig_block['y_2'])

        im = cv2.imread(image_path)
        figure = im[int(y_1):int(y_2),int(x_1):int(x_2)]

        if not os.path.exists(f"{figures_path}/{object_type}/untitled_{object_type}"):
            os.makedirs(f"{figures_path}/{object_type}/untitled_{object_type}")
        
        # ???????????????? ???????????? ????????????????????????
        list_images = os.listdir(f"{figures_path}/{object_type}/untitled_{object_type}")
        # ?????????????????? ???????????? ?????????? ???????????????? ?????????????????? ??????????????
        list_images.sort()
        
        # ???????????? ????????????????
        image_name_pattern = "untitled_"
        # ?????????????????? ?????????????????????? ?????????????????????????????? ??????????????
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

        cv2.imwrite(f"{figures_path}/{object_type}/untitled_{object_type}/{image_name}.png", figure)
    
    # TABLES
    def get_tables_with_info(self,layout_blocks_sorted,page_path,figures_path):
            blocks_volume = len(layout_blocks_sorted)
            for block_num in range(blocks_volume):
                if layout_blocks_sorted[block_num]['type'] == "Table":
                    y_1, y_2, x_1, x_2 = get_coordinates(layout_blocks_sorted[block_num])
                    page_image = cv2.imread(page_path)

                    page_width = page_image.shape[1]
                    page_height = page_image.shape[0]
                    figure_width = x_2-x_1
                    fw_to_iw = figure_width/page_width
                    fh_to_ph = (y_2-y_1)/page_height

                    # ?????????????????????? ???? ?????? ????????????????
                    if (fh_to_ph > 0.785) and (fw_to_iw > 0.60):
                        fig_union_image = cv2.rotate(page_image, cv2.cv2.ROTATE_90_CLOCKWISE)
                        # ???????????????????????????? ??????????????????????
                        figure = page_image[int(y_1):int(y_2),int(x_1):int(x_2)]
                        figure = cv2.rotate(figure, cv2.cv2.ROTATE_90_CLOCKWISE)
                    # ?????????????????????? ???? ???? ?????? ????????????????
                    else:
                        figure = page_image[int(y_1):int(y_2),int(x_1):int(x_2)]
                        if block_num >= 2:
                            y_1_check, y_2_check, _, _ = get_coordinates(layout_blocks_sorted[block_num-2])
                            if (y_2_check - y_1_check)/page_image.shape[0] < 0.017:
                                y_1_union, y_2_union, x_1_union, x_2_union = get_union_cooridnates(layout_blocks_sorted[block_num-2],layout_blocks_sorted[block_num])
                            else:
                                y_1_union, y_2_union, x_1_union, x_2_union = get_union_cooridnates(layout_blocks_sorted[block_num-1],layout_blocks_sorted[block_num])
                        elif block_num >= 1:
                                y_1_union, y_2_union, x_1_union, x_2_union = get_union_cooridnates(layout_blocks_sorted[block_num-1],layout_blocks_sorted[block_num])
                        else:
                            y_1_union, y_2_union, x_1_union, x_2_union = y_1, y_2, x_1, x_2
                        fig_union_image = page_image[y_1_union:y_2_union,x_1_union:x_2_union]
                    try:
                        figure_text = self.ocr_agent.detect(fig_union_image)
                        result = re.search(self.pattern_table,figure_text)
                        fig_num = result.group(0)
                        fig_name = fig_num.replace(" ","_").replace(".","").lower()
                        fig_name = fig_name.strip("\n")
                        if not os.path.exists(f"{figures_path}/tables"):
                            os.makedirs(f"{figures_path}/tables")

                        cv2.imwrite(f"{figures_path}/tables/{fig_name}.png", figure)
                        self.write_to_json(f"{figures_path}/tables",fig_name,object_type ='table')
                        description = re.search(self.pattern_table_desc,figure_text)
                        description = description.group(0)
                        description = description.replace(fig_num,"")
                        description = description.strip("\n\f")
                        description = description.strip("\n\x0c")
                        description = description.strip("\n")
                        # description = description[2:]
                        self.write_to_json(f"{figures_path}/tables",fig_name,item_description=description,object_type ='table')
                    except AttributeError:
                        self.save_image_as_it_is(layout_blocks_sorted[block_num],page_path,
                                    figures_path,object_type="tables")

    def extract_figures_from_a_single_pdf_file(self,path_pdf,path_out,image_format="PNG"):
        temp_path = f"{path_out}/temp"
        if os.path.exists(temp_path) == False:
                os.makedirs(temp_path)

        # ?????????????????????? pdf-?????????? ?? ??????????????????????
        convert_from_path(path_pdf,500,output_folder=temp_path,
                                    fmt=image_format.lower()
                                    )
        pages_list = os.listdir(temp_path)
        pages_list.sort()
        for page in pages_list:
            image_path = os.path.join(temp_path,page)
            image = cv2.imread(image_path)
            layout = self.model.detect(image)            
            try:
                layout = self.model.detect(image)
                layout_blocks_sorted = sort_layout_blocks(layout.to_dict()['blocks'])
            except Exception as e:
                self.logger(e,image_path,path_out)
                cv2.imwrite(f"{path_out}/logs/{page}",image)
                pass
            try:
                self.get_figures_with_info(layout_blocks_sorted,image_path,path_out)
                # self.save_figures_from_the_page(layout,image_path,path_out)
            except Exception as e:
                self.logger(e,image_path,path_out)
                cv2.imwrite(f"{path_out}/logs/{page}",image)
                pass
            try:
                self.get_tables_with_info(layout_blocks_sorted,image_path,path_out)
                # self.save_figures_from_the_page(layout,image_path,path_out)
            except Exception as e:
                self.logger(e,image_path,path_out)
                cv2.imwrite(f"{path_out}/logs/{page}",image)
                pass

    # ???? ???????????? pdf-???????????? ???????????????????? ?????????? ?? ??????????????????????????
    # ?????????????????????????? ?? ??????????????????
    def extract_figures_from_pdf_files(self,path_in,path_out,image_format="PNG"):
    # ?????????????? ???????????? ????????????
        file_list = os.listdir(path_in)
        file_list.sort()

        for item in tqdm(file_list):
            file_path = os.path.join(path_in,item)
        # ???????????????? ?????????????????????? ?? ?????????? ?? ???????????? ?????????? ????
        # ?????? ?? ?? ?????????? (?????? ????????????????????)
            paper_name = item[:-4]
            figures_path = f'{path_out}/{paper_name}'

            if os.path.exists(figures_path) == False:
                os.makedirs(figures_path)
            self.extract_figures_from_a_single_pdf_file(file_path,figures_path,image_format)