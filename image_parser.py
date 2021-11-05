import os
import cv2
import layoutparser as lp
from pdf2image import convert_from_path
import PIL
import matplotlib.pyplot as plt
import re

from tqdm import tqdm
import numpy as np

import sys
import argparse


class ImageParser():
    def __init__(self):
        self.model = model = lp.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.80],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
        self.pattern_fig = "Fig. \d*|Figure \d*|Scheme \d*"
        self.ocr_agent = lp.TesseractAgent(languages='eng')

    # вытаскиваем описание описание строго снизу
    def get_fig_n_below(self,fig_block,image_path,figures_path,paper_name):
        x_1,y_1,x_2,y_2 = np.floor(fig_block['x_1']),np.floor(fig_block['y_1']),np.ceil(fig_block['x_2']),np.ceil(fig_block['y_2'])

        im = cv2.imread(image_path)
        page_width = im.shape[1]
        figure_width = x_2-x_1
        fw_to_iw = figure_width/page_width

        fh_to_ph = (y_2-y_1)/im.shape[0]

        # изображения на всю страницу
        if (fh_to_ph > 0.84) and (fw_to_iw > 0.59):
            im_desc = cv2.rotate(im, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            # переворачиваем изображение
            im = im[int(y_1):int(y_2),int(x_1):int(x_2)]
            im = cv2.rotate(im, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

        # изображение не на всю страницу
        else:
            # доля от высоты изображения, на которую мы будем спускаться в поисках номера изображения
            y_percent = 0.125
            if fh_to_ph <0.2:
                y_percent = 0.19
            elif fh_to_ph>=0.6:
                y_percent = 0.08

            # если изображение больше, чем в два столбца текста
            if fw_to_iw >=0.485:
                delta_x = int(x_1*0.95)
                delta_y = int((y_2-y_1)*y_percent)

            # если изображение помещается в один столбец текста
            else:
                # если изображение находится в правом столбце
                if page_width - x_2> x_1:
                    delta_x = int(x_1*0.48)
                # если изображение в столбце слева
                else:
                    delta_x = int(page_width*(0.49 -fw_to_iw )/2)
                delta_y = int((y_2-y_1)*y_percent)
            
            

        # im = im[int(y_1):int(y_2)+delta_y,int(x_1)-delta_x:int(x_2)]
            im_desc = im[int(y_2):int(y_2)+delta_y,int(x_1)-delta_x:int(x_2)]

            im = im[int(y_1):int(y_2),int(x_1):int(x_2)]

        figure_text = self.ocr_agent.detect(im_desc)
        result = re.search(self.pattern_fig,figure_text)
        fig_num = result.group(0)


        fig_name = fig_num.replace(" ","_").replace(".","").lower()

        cv2.imwrite(f"{figures_path}/{fig_name}.png", im)

    # вытаскиваем описание описание справа или слева от изображения
    def get_fig_n_sides(self,fig_block,image_path,figures_path,paper_name):
        x_1,y_1,x_2,y_2 = np.floor(fig_block['x_1']),np.floor(fig_block['y_1']),np.ceil(fig_block['x_2']),np.ceil(fig_block['y_2'])

        im = cv2.imread(image_path)
        page_width = im.shape[1]
        figure_width = x_2-x_1
        fw_to_iw = figure_width/page_width

        if page_width-x_2 > x_1:
            # описание находится справа
            # delta_x = int((page_width-x_2)*0.95)
            delta_x = int((page_width-x_2)*0.35)
            delta_y = int((y_2-y_1)*0.02)
            im_desc = im[int(y_1)-delta_y:int(y_2)+delta_y,int(x_1):int(x_2)+delta_x]
        else:
            # описание нахоидтся слева
            delta_x = int(x_1*0.95)
            delta_y = int((y_2-y_1)*0.02)
            im_desc = im[int(y_1)-delta_y:int(y_2)+delta_y,int(x_1)-delta_x:int(x_2)]

        # распознаём текст описания
        figure_text = self.ocr_agent.detect(im)
        result = re.search(self.pattern_fig,figure_text)
        fig_num = result.group(0)


        fig_name = fig_num.replace(" ","_").replace(".","").lower()


        cv2.imwrite(f"{figures_path}/{fig_name}.png", im)

    def save_image_as_it_is(self,fig_block,image_path,figures_path,paper_name):
        x_1,y_1,x_2,y_2 = np.floor(fig_block['x_1']),np.floor(fig_block['y_1']),np.ceil(fig_block['x_2']),np.ceil(fig_block['y_2'])

        im = cv2.imread(image_path)
        im = im[int(y_1):int(y_2),int(x_1):int(x_2)]

        # if not os.path.exists(f"/out/{paper_name}/untitled_images"):
        if not os.path.exists(f"{figures_path}/untitled_images"):
            os.makedirs(f"{figures_path}/untitled_images")
        
        # получаем список изображений
        # list_images = os.listdir(f"/out/{paper_name}/untitled_images")
        list_images = os.listdir(f"{figures_path}/untitled_images")
        # сортируем списко чтобы получить последний элемент
        list_images.sort()
        
        # шаблон названия
        image_name_pattern = "untitled_"
        # оставляем изображения соответствующие шаблону
        list_images = [el for el in list_images if image_name_pattern in el]

        if len(list_images)==0:
            # первое изображение без названия
            image_name = "untitled_0"
        else:
            # получаем имя последнего изображения без расширения
            image_name = list_images[-1][:-4]
            # получаем номер изображения
            image_number = image_name.replace(image_name_pattern,"")
            image_number = int(image_number)+1
            # имя n-го изображения
            image_name = image_name_pattern + str(image_number)

        # cv2.imwrite(f"/out/{paper_name}/untitled_images/{image_name}.png", im)
        cv2.imwrite(f"{figures_path}/untitled_images/{image_name}.png", im)

    # сохраняет предоставленное изображение
    # в случае наличия номера в виде Fig. n или Figure n сохраняет в виде 
    def save_figure_with_number(self,fig_block,image_path,figures_path,paper_name):
        # предположим, что у изображения есть описание
        try:
            # Допустим описание к изображению находится снизу
            try:
                self.get_fig_n_below(self,fig_block,image_path,figures_path,paper_name)

            # допустим описание к изображению находится справа или слева   
            except AttributeError:
                self.get_fig_n_sides(self,fig_block,image_path,figures_path,paper_name)
        # изображения без описания
        except AttributeError:
            self.save_image_as_it_is(self,fig_block,image_path,figures_path,paper_name)

    def save_figures_from_the_page(self,layout,image_path,figures_path,paper_name):

        for block in layout.to_dict()['blocks']:
            if block['type']=="Figure":
                self.save_figure_with_number(self,block,image_path,figures_path,paper_name)

    def extract_figures_from_pdf_files(self,path_in,path_out,image_format="PNG"):

    # получим список файлов
        file_list = os.listdir(path_in)
        file_list.sort()

        for item in tqdm(file_list):
            file_path = os.path.join(path_in,item)
        # поместим изображение в папку с именем таким же
        # как и у файла (без расширения)
            paper_name = item[:-4]
            figures_path = f'{path_out}/{paper_name}'

            if os.path.exists(figures_path)==False:
                os.makedirs(figures_path)


        # конвертация pdf-файла в изображение
            pages = convert_from_path(file_path,500)
            for n,page in enumerate(pages,1):
                image_path = f'{figures_path}/{n}.png'
                page.save(image_path,image_format)
                image = cv2.imread(image_path)
                layout = self.model.detect(image)
                self.save_figures_from_the_page(layout,image_path,figures_path,paper_name)
                os.remove(image_path)




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
