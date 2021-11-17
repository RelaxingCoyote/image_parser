from abc import abstractmethod
import os
import sys
import argparse
import re
import json
from tqdm import tqdm
import numpy as np
import cv2
import layoutparser as lp
from pdf2image import convert_from_path

from warnings import filterwarnings

filterwarnings('ignore')

# принимает две точки, возвращает расстояние между точками
def calculate_distance(x,x_,y,y_):
    return np.sqrt((x + x_)**2 + (y + y_)**2)

# принимает две пары точек, возвращает сумму расстояний между двумя парами точек
def calculate_double_distance(pair_of_coordinates_0, pair_of_coordinates_1):
    x_0,x_0_,y_0,y_0_ = pair_of_coordinates_0
    x_1,x_1_,y_1,y_1_ = pair_of_coordinates_1
    return calculate_distance(x_0,x_0_,y_0,y_0_) + calculate_distance(x_1,x_1_,y_1,y_1_)

class ImageParser():

    def __init__(self):
        self.model = lp.Detectron2LayoutModel(
                                                'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.80],
                                                label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
                                                )
        self.mfd_model = lp.Detectron2LayoutModel('lp://MFD/faster_rcnn_R_50_FPN_3x/config',
                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.79], label_map={1: "Equation"})                                                     
        self.pattern_fig = re.compile("Fig. \d*|Figure \d*|Scheme \d*",re.IGNORECASE)
        self.pattern_table = re.compile("Table \d*[^.,]",re.IGNORECASE)
        self.pattern_fig_desc = re.compile("Fig. \d*[\.\:] [A-Z][\s\S]+|"
                                            "Figure \d*[\.\:] [A-Z][\s\S]+|Scheme \d* [A-Z][\.\:][\s\S]+",re.IGNORECASE)
        self.pattern_table_desc = re.compile("Table \d*[\s\S]+?(?=\n\n)",re.IGNORECASE)
        self.pattern_formula_desc = re.compile("\(\S{1,5}\)")

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

    def read_delete_temp(self,file_directory,object_type="figures"):
        temp_path = os.path.join(file_directory,object_type,"temp.json")
        if os.path.exists(temp_path):
            with open(temp_path, "r") as read_file:
                fig_dict = json.load(read_file)
            # fig_dict = sorted()
            return fig_dict[object_type]

    # Метод, извлекающий описание к изображению или таблице
    def get_fig_desc(self,text_block,image_path):
        x_1,y_1 = np.floor(text_block['x_1']),np.floor(text_block['y_1'])
        x_2,y_2 = np.ceil(text_block['x_2']),np.ceil(text_block['y_2'])

        im = cv2.imread(image_path)
        text_image = im[int(y_1):int(y_2),int(x_1):int(x_2)]

        text = self.ocr_agent.detect(text_image)
        result = re.search(self.pattern_fig_desc,text)
        fig_desc = result.group(0)
        return fig_desc

    def save_description_from_the_page(self,layout,image_path,out_path):
        figures_path = f"{out_path}/figures"
        if os.path.exists(figures_path):
            for block in layout.to_dict()['blocks']:
                if block['type'] == "Text" or block['type'] == "Title":
                    try:
                        fig_desc = self.get_fig_desc(block,image_path)
                        result = re.search(self.pattern_fig,fig_desc)
                        fig_num = result.group(0)

                        fig_id = fig_num.replace(" ","_").replace(".","").lower()
                        if os.path.exists(os.path.join(figures_path,"temp.json")):
                            with open(os.path.join(figures_path,"temp.json"), "r") as read_file:
                                fig_dict = json.load(read_file)
                            for count in range(len(fig_dict['figures'])):   
                                if fig_id == fig_dict['figures'][count]['figure_id']:
                                    # убираем номер изображения из строки описания
                                    fig_desc = fig_desc.replace(fig_num,"")
                                    fig_desc = fig_desc.strip("\n\f")
                                    fig_desc = fig_desc[2:]
                                    fig_dict['figures'][count]['describe'] = fig_desc
                                    # self.write_to_json(self,figures_path,fig_num,fig_desc)
                                with open(os.path.join(figures_path,"temp.json"), "w") as write_file:
                                    json.dump(fig_dict, write_file) 

                    except AttributeError:
                        pass

    
    # Вытаскиваем описание описание строго снизу
    def get_fig_n_below(self,fig_block,image_path,figures_path):
        # Получим координаты изображения
        x_1,y_1 = np.floor(fig_block['x_1']),np.floor(fig_block['y_1'])
        x_2,y_2 = np.ceil(fig_block['x_2']),np.ceil(fig_block['y_2'])

        im = cv2.imread(image_path)
        # Получаем отношение ширины изображения к ширине страницы,
        # а также отношение высототы изображения к высоте страницы
        page_width = im.shape[1]
        figure_width = x_2-x_1
        fw_to_iw = figure_width/page_width
        fh_to_ph = (y_2-y_1)/im.shape[0]

        # Изображения на всю страницу
        if (fh_to_ph > 0.785) and (fw_to_iw > 0.60):
            im_desc = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)
            # Переворачиваем изображение
            figure = im[int(y_1):int(y_2),int(x_1):int(x_2)]
            figure = cv2.rotate(figure, cv2.cv2.ROTATE_90_CLOCKWISE)
        # Изображение не на всю страницу
        else:
            # Доля от высоты изображения, на которую мы будем спускаться в поисках номера изображения
            y_percent = 0.125
            if fh_to_ph < 0.2:
                y_percent = 0.19
            elif fh_to_ph >= 0.6:
                y_percent = 0.08

            # Если изображение в два столбца текста
            if fw_to_iw >= 0.485:
                delta_x = int(x_1*0.95)
                delta_y = int((y_2-y_1)*y_percent)

            # Если изображение помещается в один столбец текста
            else:
                # Если изображение находится в правом столбце
                if page_width - x_2> x_1:
                    delta_x = int(x_1*0.48)
                # Если изображение в столбце слева
                else:
                    delta_x = int(page_width*(0.49 - fw_to_iw )/2)
                delta_y = int((y_2-y_1)*y_percent)
            
            im_desc = im[int(y_2):int(y_2)+delta_y,int(x_1)-delta_x:int(x_2)]

            figure = im[int(y_1):int(y_2),int(x_1):int(x_2)]

        figure_text = self.ocr_agent.detect(im_desc)
        result = re.search(self.pattern_fig,figure_text)
        fig_num = result.group(0)

        fig_name = fig_num.replace(" ","_").replace(".","").lower()
        if not os.path.exists(f"{figures_path}/figures"):
            os.makedirs(f"{figures_path}/figures")

        cv2.imwrite(f"{figures_path}/figures/{fig_name}.png", figure)
        self.write_to_json(f"{figures_path}/figures",fig_name,object_type ='figure')             

    # Вытаскиваем описание описание справа или слева от изображения
    def get_fig_n_sides(self,fig_block,image_path,figures_path):
        # Получим координаты изображения
        x_1,y_1 = np.floor(fig_block['x_1']),np.floor(fig_block['y_1'])
        x_2,y_2 = np.ceil(fig_block['x_2']),np.ceil(fig_block['y_2'])

        im = cv2.imread(image_path)
        page_width = im.shape[1]
        figure_width = x_2 - x_1
        fw_to_iw = figure_width/page_width

        if page_width-x_2 > x_1:
            # Описание находится справа
            delta_x = int((page_width - x_2)*0.35)
            delta_y = int((y_2 - y_1)*0.02)
            im_desc = im[int(y_1)-delta_y:int(y_2)+delta_y,int(x_1):int(x_2)+delta_x]
        else:
            # Описание нахоидтся слева
            delta_x = int(x_1*0.95)
            delta_y = int((y_2 - y_1)*0.02)
            im_desc = im[int(y_1)-delta_y:int(y_2)+delta_y,int(x_1)-delta_x:int(x_2)]

        # Распознаём текст описания
        figure_text = self.ocr_agent.detect(im_desc)
        result = re.search(self.pattern_fig,figure_text)
        fig_num = result.group(0)


        fig_name = fig_num.replace(" ","_").replace(".","").lower()

        if not os.path.exists(f"{figures_path}/figures"):
            os.makedirs(f"{figures_path}/figures")

        figure = im[int(y_1):int(y_2),int(x_1):int(x_2)]
        cv2.imwrite(f"{figures_path}/figures/{fig_name}.png", figure)

        self.write_to_json(f"{figures_path}/figures",fig_name,object_type ='figure') 

    # Сохраняет изображение как undefined_n
    def save_image_as_it_is(self,fig_block,image_path,
                            figures_path,object_type="figures"):
        # Получим координаты изображения
        x_1,y_1 = np.floor(fig_block['x_1']),np.floor(fig_block['y_1'])
        x_2,y_2 = np.ceil(fig_block['x_2']),np.ceil(fig_block['y_2'])

        im = cv2.imread(image_path)
        figure = im[int(y_1):int(y_2),int(x_1):int(x_2)]

        if not os.path.exists(f"{figures_path}/{object_type}/untitled_{object_type}"):
            os.makedirs(f"{figures_path}/{object_type}/untitled_{object_type}")
        
        # Получаем список изображенийЫ
        list_images = os.listdir(f"{figures_path}/{object_type}/untitled_{object_type}")
        # Сортируем списко чтобы получить последний элемент
        list_images.sort()
        
        # Шаблон названия
        image_name_pattern = "untitled_"
        # Оставляем изображения соответствующие шаблону
        list_images = [el for el in list_images if image_name_pattern in el]

        if len(list_images)==0:
            # Первое изображение без названия
            image_name = "untitled_0"
        else:
            # Получаем имя последнего изображения без расширения
            image_name = list_images[-1][:-4]
            # Получаем номер изображения
            image_number = image_name.replace(image_name_pattern,"")
            image_number = int(image_number) + 1
            # Имя n-го изображения
            image_name = image_name_pattern + str(image_number)

        cv2.imwrite(f"{figures_path}/{object_type}/untitled_{object_type}/{image_name}.png", figure)
    
    # Сохраняет предоставленное изображение
    # В случае наличия номера в виде Fig. n, Figure n или Scheme n сохраняет в виде
    # fig_n, figure_n или scheme_n
    def save_figure_with_number(self,fig_block,image_path,figures_path):
        # Предположим, что у изображения есть описание
        try:
            # Допустим описание к изображению находится снизу
            try:
                self.get_fig_n_below(fig_block,image_path,figures_path)

            # Допустим описание к изображению находится справа или слева   
            except AttributeError:
                self.get_fig_n_sides(fig_block,image_path,figures_path)
        # Изображения без описания
        except AttributeError:
            self.save_image_as_it_is(fig_block,image_path,figures_path)

    def get_the_closest_block_beneath(self,layout,coordinate_list):
        y_1, y_2, x_1, x_2 = coordinate_list

        closest_block_beneath = None
        closest_block_beneath_distance = np.inf()
        for block in layout.to_dict()['blocks']:
            if block['type'] == "Text" or block['type'] == "Title":
                y_1_cbb, y_2_cbb, x_1_cbb, x_2_cbb = self.get_coordinates(block)
                if y_1_cbb < y_2:
                    dot_pair_0 = [x_1,x_1_cbb,y_2,y_1_cbb]
                    dot_pair_1 = [x_2,x_2_cbb,y_2,y_1_cbb]
                    closest_block_beneath_distance_new = calculate_double_distance(dot_pair_0,dot_pair_1)
                    if closest_block_beneath_distance_new < closest_block_beneath_distance:
                        closest_block_beneath_distance = closest_block_beneath_distance_new
                        closest_block_beneath = block

        return closest_block_beneath
    
    # Вытаскиваем описание описание строго снизу
    def get_table_n(self,fig_block,image_path,figures_path):
        # Получим координаты таблицы
        x_1,y_1 = np.floor(fig_block['x_1']),np.floor(fig_block['y_1'])
        x_2,y_2 = np.ceil(fig_block['x_2']),np.ceil(fig_block['y_2'])

        im = cv2.imread(image_path)
        # Получаем отношение ширины таблицы к ширине страницы,
        # а также отношение высототы таблицы к высоте страницы
        page_width = im.shape[1]
        figure_width = x_2-x_1
        fw_to_iw = figure_width/page_width
        fh_to_ph = (y_2-y_1)/im.shape[0]

        # Таблица на всю страницу
        if (fh_to_ph > 0.785) and (fw_to_iw > 0.60):
            im_desc = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)
            # Переворачиваем таблицу
            im = im[int(y_1):int(y_2),int(x_1):int(x_2)]
            im = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)

        # Таблица не на всю страницу
        else:
            # Доля от высоты таблицы, на которую мы будем спускаться в поисках номера таблицы
            y_percent = 0.25
            if fh_to_ph < 0.2 and fh_to_ph < 0.15:
                y_percent = 0.79
            elif fh_to_ph < 0.15 and fh_to_ph > 0.10:
                y_percent = 0.87
            elif fh_to_ph < 0.10:
                y_percent = 0.98
            elif fh_to_ph>=0.4 and fh_to_ph<0.7:
                y_percent = 0.40
            elif fh_to_ph>=0.7:
                y_percent = 0.15

            # Если таблица больше, чем в два столбца текста
            if fw_to_iw >= 0.485:
                delta_x = int(x_1*0.95)
                delta_y = int((y_2-y_1)*y_percent)

            # Если таблица помещается в один столбец текста
            else:
                # Если таблица находится в правом столбце
                if page_width - x_2> x_1:
                    delta_x = int(x_1*0.48)
                # Если таблица в столбце слева
                else:
                    delta_x = int(page_width*(0.49 - fw_to_iw )/2)
                delta_y = int((y_2-y_1)*y_percent)
            
            
            im_desc = im[int(y_1)-delta_y:int(y_1),int(x_1)-delta_x:int(x_2)]

            im = im[int(y_1):int(y_2),int(x_1):int(x_2)]

        table_text = self.ocr_agent.detect(im_desc)
        result = re.search(self.pattern_table,table_text)
        table_num = result.group(0)

        table_id = table_num.replace(" ","_").replace(".","").lower()
        table_id = table_id.strip('\n')
        if not os.path.exists(f"{figures_path}/tables"):
            os.makedirs(f"{figures_path}/tables")

        cv2.imwrite(f"{figures_path}/tables/{table_id}.png", im)

        self.write_to_json(f"{figures_path}/tables",table_id,item_description='',object_type='table')

        try:
            self.pattern_table_desc
            self.get_desc(table_id,table_num,table_text,self.pattern_table_desc,figures_path)
        except AttributeError:
            pass

    # Метод, извлекающий описание к изображению или таблице
    def get_desc(self,table_id,table_num,table_text,pattern_desc,figures_path):
        result = re.search(pattern_desc,table_text)
        table_description = result.group(0)
        table_description = table_description.strip(table_num).strip('\n')

        sub_branch = 'table'

        self.write_to_json(f"{figures_path}/tables",table_id,table_description,object_type=sub_branch)

    # Сохраняет предоставленную таблицу
    # В случае наличия номера в виде Table n
    # table_n
    def save_table_with_number(self,fig_block,image_path,figures_path):
        # Предположим, что у таблицы  есть описание
        try:
            # Допустим описание к таблице находится сверху
            # Можно добавить в вайл temp.json название статьи
            self.get_table_n(fig_block,image_path,figures_path)
        # Таблица без описания
        except AttributeError:
            self.save_image_as_it_is(fig_block,image_path,
                                        figures_path,object_type="tables")

    # Formulas
    def get_formula(self,block,image_path,formula_path):
        # Получим координаты формулы
        x_1,y_1 = np.floor(block['x_1']),np.floor(block['y_1'])
        x_2,y_2 = np.ceil(block['x_2']),np.ceil(block['y_2'])

        im = cv2.imread(image_path)

        page_hight = im.shape[0]
        page_width = im.shape[1]

        delta_x = int((page_width - x_2)*0.95)
        delta_y = int(page_hight*0.05)

        im_desc = im[int(y_1)+delta_y:int(y_2)+delta_y,int(x_1):int(x_2)+delta_x]

        formula_text = self.ocr_agent.detect(im_desc)
        result = re.search(self.pattern_formula_desc,formula_text)
        im = im[int(y_1):int(y_2),int(x_1):int(x_2)]

        formula_num = result.group(0)
        formula_num.replace("(","_").replace(")","").lower()

        if not os.path.exists(f"{formula_path}"):
            os.makedirs(f"{formula_path}")

        cv2.imwrite(f"{formula_path}/formula_{formula_num}", im)

        self.write_to_json(f"{formula_path}","formula_{formula_num}.png",item_description='',object_type='formula')

    # Сохраняет изображения (таблицы) со страницы документа
    def save_figures_from_the_page(self,layout,image_path,figures_path):
        for block in layout.to_dict()['blocks']:
            if block['type'] == "Figure":
                self.save_figure_with_number(block,image_path,figures_path)
            if block['type'] == "Table":
                self.save_table_with_number(block,image_path,figures_path)

    def save_formula_with_number(self,fig_block,image_path,formula_path):
        # Предположим, что у formula есть описание
        try:
            self.get_formula(fig_block,image_path,formula_path)

        # Изображения без описания
        except AttributeError:
            self.save_image_as_it_is(fig_block,image_path,formula_path,object_type="")

    # Сохраняет формулы со страницы документа
    def save_formulas_from_the_page(self,layout,image_path,formula_path):
        formula_path = os.path.join(formula_path,"formulas")
        for block in layout.to_dict()['blocks']:
            if block['type'] == "Equation":
                self.save_formula_with_number(block,image_path,formula_path)

    # # Сохраняет предоставленную таблицу
    # # В случае наличия номера в виде Table n
    # # table_n
    # def save_table_with_number(self,fig_block,image_path,figures_path,paper_name):
    #     # Предположим, что у таблицы  есть описание
    #     try:
    #         # Допустим описание к таблице находится сверху
    #         self.get_table_n(fig_block,image_path,figures_path,paper_name)
    #     # Таблица без описания
    #     except AttributeError:
    #         self.save_image_as_it_is(fig_block,image_path,figures_path,paper_name,"tables")


    def extract_figures_from_a_single_pdf_file(self,path_pdf,path_out,image_format="PNG"):
        temp_path = f"{path_out}/temp"
        if os.path.exists(temp_path) == False:
                os.makedirs(temp_path)

        # Конвертация pdf-файла в изображение
        convert_from_path(path_pdf,500,output_folder=temp_path,
                                    fmt=image_format.lower()
                                    )
        pages_list = os.listdir(temp_path)
        pages_list.sort()
        for page in pages_list:
            image_path = os.path.join(temp_path,page)
            image = cv2.imread(image_path)
            layout = self.model.detect(image)            
            # try:
            #     layout = self.model.detect(image)
            # except Exception as e:
            #     self.logger(e,image_path,path_out)
            #     cv2.imwrite(f"{path_out}/logs/{page}",image)
            #     pass
            # try:
            #     self.save_figures_from_the_page(layout,image_path,path_out)
            # except Exception as e:
            #     self.logger(e,image_path,path_out)
            #     cv2.imwrite(f"{path_out}/logs/{page}",image)
            #     pass
            # try:
            #     self.save_description_from_the_page(layout,image_path,path_out)
            # except Exception as e:
            #     self.logger(e,image_path,path_out)
            #     cv2.imwrite(f"{path_out}/logs/{page}",image)
            #     pass

            layout = self.model.detect(image)
            self.save_figures_from_the_page(layout,image_path,path_out)
            self.save_description_from_the_page(layout,image_path,path_out)
            layout = self.mfd_model.detect(image)
            self.save_formulas_from_the_page(layout,image_path,path_out)

        #     os.remove(image_path)
        # os.rmdir(temp_path)

        for root, dirs, files in os.walk(temp_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            # for name in dirs:
            #     os.rmdir(os.path.join(root, name))
        os.rmdir(temp_path)

    # Из набора pdf-файлов генерирует папки с изображениями
    # содержащимися в документе
    def extract_figures_from_pdf_files(self,path_in,path_out,image_format="PNG"):
    # Получим список файлов
        file_list = os.listdir(path_in)
        file_list.sort()

        for item in tqdm(file_list):
            file_path = os.path.join(path_in,item)
        # Поместим изображение в папку с именем таким же
        # Как и у файла (без расширения)
            paper_name = item[:-4]
            figures_path = f'{path_out}/{paper_name}'

            if os.path.exists(figures_path) == False:
                os.makedirs(figures_path)
            self.extract_figures_from_a_single_pdf_file(file_path,figures_path,image_format)