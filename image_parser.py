import os
import sys
import argparse
import re
from tqdm import tqdm
import numpy as np
import cv2
import layoutparser as lp
from pdf2image import convert_from_path

from warnings import filterwarnings

filterwarnings('ignore')


class ImageParser():


    def __init__(self):
        self.model = lp.Detectron2LayoutModel(
                                                'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.80],
                                                label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
                                                )                                                     
        self.pattern_fig = re.compile("Fig. \d*|Figure \d*|Scheme \d*",re.IGNORECASE)
        self.pattern_table = re.compile("Table \d*[^.,]",re.IGNORECASE)
        self.pattern_fig_desc = re.compile("Fig. \d*[\s\S]+?(?=\n\n)\
                                            |Figure \d*[\s\S]+?(?=\n\n)|Scheme \d*[\s\S]+?(?=\n\n)",re.IGNORECASE)
        self.pattern_table_desc = re.compile("Table \d*[\s\S]+?(?=\n\n)",re.IGNORECASE)

        self.ocr_agent = lp.TesseractAgent(languages='eng')

    def logger(self,what_happened,where_happened,path_out):
        if not os.path.exists(f"{path_out}/log.txt"):
            f = open(f"{path_out}/logs/log.txt","w")
        else:
            f = open(f"{path_out}/logs/log.txt","a")
        f.write(f"{what_happened} : {where_happened}\n")

    # Метод, извлекающий описание к изображению или таблице
    def get_desc(self,figure_text):
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

    # # Вытаскиваем описание описание строго снизу
    # def get_table_n(self,fig_block,image_path,figures_path,paper_name):
    #     # Получим координаты таблицы
    #     x_1,y_1 = np.floor(fig_block['x_1']),np.floor(fig_block['y_1'])
    #     x_2,y_2 = np.ceil(fig_block['x_2']),np.ceil(fig_block['y_2'])

    #     im = cv2.imread(image_path)
    #     # Получаем отношение ширины таблицы к ширине страницы,
    #     # а также отношение высототы таблицы к высоте страницы
    #     page_width = im.shape[1]
    #     figure_width = x_2-x_1
    #     fw_to_iw = figure_width/page_width
    #     fh_to_ph = (y_2-y_1)/im.shape[0]

    #     # Таблица на всю страницу
    #     if (fh_to_ph > 0.785) and (fw_to_iw > 0.60):
    #         im_desc = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)
    #         # Переворачиваем таблицу
    #         im = im[int(y_1):int(y_2),int(x_1):int(x_2)]
    #         im = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)

    #     # Таблица не на всю страницу
    #     else:
    #         # Доля от высоты таблицы, на которую мы будем спускаться в поисках номера таблицы
    #         y_percent = 0.25
    #         if fh_to_ph < 0.2 and fh_to_ph < 0.15:
    #             y_percent = 0.75
    #         elif fh_to_ph < 0.15 and fh_to_ph > 0.10:
    #             y_percent = 0.60
    #         elif fh_to_ph < 0.10:
    #             y_percent = 0.90
    #         elif fh_to_ph>=0.6:
    #             y_percent = 0.15

    #         # Если таблица больше, чем в два столбца текста
    #         if fw_to_iw >= 0.485:
    #             delta_x = int(x_1*0.95)
    #             delta_y = int((y_2-y_1)*y_percent)

    #         # Если таблица помещается в один столбец текста
    #         else:
    #             # Если таблица находится в правом столбце
    #             if page_width - x_2> x_1:
    #                 delta_x = int(x_1*0.48)
    #             # Если таблица в столбце слева
    #             else:
    #                 delta_x = int(page_width*(0.49 - fw_to_iw )/2)
    #             delta_y = int((y_2-y_1)*y_percent)
            
            
    #         im_desc = im[int(y_1)-delta_y:int(y_1),int(x_1)-delta_x:int(x_2)]

    #         im = im[int(y_1):int(y_2),int(x_1):int(x_2)]

    #     table_text = self.ocr_agent.detect(im_desc)
    #     result = re.search(self.pattern_table,table_text)
    #     table_num = result.group(0)


    #     table_name = table_num.replace(" ","_").replace(".","").lower()
    #     if not os.path.exists(f"{figures_path}/tables"):
    #         os.makedirs(f"{figures_path}/tables")

    #     cv2.imwrite(f"{figures_path}/tables/{table_name}.png", im)

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

    # Сохраняет изображения (таблицы) со страницы документа
    def save_figures_from_the_page(self,layout,image_path,figures_path):
        for block in layout.to_dict()['blocks']:
            if block['type'] == "Figure":
                self.save_figure_with_number(block,image_path,figures_path)
            # if block['type'] == "Table":
            #     self.save_table_with_number(block,image_path,figures_path,paper_name)

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
            try:
                layout = self.model.detect(image)
            except Exception as e:
                self.logger(e,image_path,path_out)
                cv2.imwrite(f"{path_out}/logs/{page}",image)
            try:
                self.save_figures_from_the_page(layout,image_path,path_out)
            except Exception as e:
                self.logger(e,image_path,path_out)
                cv2.imwrite(f"{path_out}/logs/{page}",image)
            # os.remove(image_path)
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
        
