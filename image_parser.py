import os
import sys
import argparse
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import numpy as np
import cv2
import layoutparser as lp
from pdf2image import convert_from_path


class ImageParser():


    def __init__(self):
        self.model = lp.Detectron2LayoutModel(
                                                        'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.80],
                                                        label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})                                                     
        self.pattern_fig = re.compile("Fig. \d*|Figure \d*|Scheme \d*",re.IGNORECASE)
        self.pattern_table = re.compile("Table \d*",re.IGNORECASE)
        self.ocr_agent = lp.TesseractAgent(languages='eng')

    # Вытаскиваем описание описание строго снизу
    def get_fig_n_below(self,fig_block,image_path,figures_path,paper_name):
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
            im = im[int(y_1):int(y_2),int(x_1):int(x_2)]
            im = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)

        # Изображение не на всю страницу
        else:
            # Доля от высоты изображения, на которую мы будем спускаться в поисках номера изображения
            y_percent = 0.125
            if fh_to_ph < 0.2:
                y_percent = 0.19
            elif fh_to_ph >= 0.6:
                y_percent = 0.08

            # Если изображение больше, чем в два столбца текста
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

            im = im[int(y_1):int(y_2),int(x_1):int(x_2)]

        figure_text = self.ocr_agent.detect(im_desc)
        result = re.search(self.pattern_fig,figure_text)
        fig_num = result.group(0)


        fig_name = fig_num.replace(" ","_").replace(".","").lower()
        if not os.path.exists(f"{figures_path}/figures"):
            os.makedirs(f"{figures_path}/figures")

        cv2.imwrite(f"{figures_path}/figures/{fig_name}.png", im)

    # Вытаскиваем описание описание справа или слева от изображения
    def get_fig_n_sides(self,fig_block,image_path,figures_path,paper_name):
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
        figure_text = self.ocr_agent.detect(im)
        result = re.search(self.pattern_fig,figure_text)
        fig_num = result.group(0)


        fig_name = fig_num.replace(" ","_").replace(".","").lower()

        if not os.path.exists(f"{figures_path}/figures"):
            os.makedirs(f"{figures_path}/figures")
        cv2.imwrite(f"{figures_path}/figures/{fig_name}.png", im)

    # Сохраняет изображение как undefined_n
    def save_image_as_it_is(self,fig_block,image_path,
                            figures_path,paper_name,object_type="figures"):
        # Получим координаты изображения
        x_1,y_1 = np.floor(fig_block['x_1']),np.floor(fig_block['y_1'])
        x_2,y_2 = np.ceil(fig_block['x_2']),np.ceil(fig_block['y_2'])

        im = cv2.imread(image_path)
        im = im[int(y_1):int(y_2),int(x_1):int(x_2)]

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

        cv2.imwrite(f"{figures_path}/{object_type}/untitled_{object_type}/{image_name}.png", im)
    
    # Сохраняет предоставленное изображение
    # В случае наличия номера в виде Fig. n, Figure n или Scheme n сохраняет в виде
    # fig_n, figure_n или scheme_n
    def save_figure_with_number(self,fig_block,image_path,figures_path,paper_name):
        # Предположим, что у изображения есть описание
        try:
            # Допустим описание к изображению находится снизу
            try:
                self.get_fig_n_below(fig_block,image_path,figures_path,paper_name)

            # Допустим описание к изображению находится справа или слева   
            except AttributeError:
                self.get_fig_n_sides(fig_block,image_path,figures_path,paper_name)
        # Изображения без описания
        except AttributeError:
            self.save_image_as_it_is(fig_block,image_path,figures_path,paper_name)

    # Вытаскиваем описание описание строго снизу
    def get_table_n(self,fig_block,image_path,figures_path,paper_name):
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
            y_percent = 0.10
            if fh_to_ph <0.2:
                y_percent = 0.40
            elif fh_to_ph>=0.6:
                y_percent = 0.08

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


        fig_name = table_num.replace(" ","_").replace(".","").lower()
        if not os.path.exists(f"{figures_path}/tables"):
            os.makedirs(f"{figures_path}/tables")

        cv2.imwrite(f"{figures_path}/tables/{table_num}.png", im)

    # Сохраняет предоставленную таблицу
    # В случае наличия номера в виде Table n
    # table_n
    def save_table_with_number(self,fig_block,image_path,figures_path,paper_name):
        # Предположим, что у таблицы  есть описание
        try:
            # Допустим описание к таблице находится сверху
            self.get_table_n(fig_block,image_path,figures_path,paper_name)
        # Таблица без описания
        except AttributeError:
            self.save_image_as_it_is(fig_block,image_path,figures_path,paper_name,"tables")

    # Сохраняет изображения (таблицы) со страницы документа
    def save_figures_from_the_page(self,layout,image_path,figures_path,paper_name):
        for block in layout.to_dict()['blocks']:
            if block['type'] == "Figure":
                self.save_figure_with_number(block,image_path,figures_path,paper_name)
            if block['type'] == "Table":
                self.save_table_with_number(block,image_path,figures_path,paper_name)

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

        # Конвертация pdf-файла в изображение
            pages = convert_from_path(file_path,500)
            for n,page in enumerate(pages,1):
                # Сохраняем страницу документа в качестве изображения
                image_path = f"{figures_path}/{n}.{image_format.lower()}"
                page.save(image_path,image_format)
                # Детектируем объекты на странице
                image = cv2.imread(image_path)
                layout = self.model.detect(image)
                # Сохраняем изображения со страницы
                self.save_figures_from_the_page(layout,image_path,figures_path,paper_name)
                # Удаляем изображение самой страницы
                os.remove(image_path)
            list_figures = os.listdir(figures_path)


# позже будет исправлен
# def createParser ():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path_in', '--path_out',action='append')
#     parser.add_argument('--path_out',default='out',action='append')
 
#     return parser
 
 
# if __name__ == '__main__':
#     parser = createParser()
#     args = vars(parser.parse_args())

#     img_parser = ImageParser()
    

#     img_parser.extract_figures_from_pdf_files(args['path_in'], args['path_out'])
