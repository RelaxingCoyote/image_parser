import numpy as np
from dataclasses import dataclass
import layoutparser as lp
from typing import ClassVar

CONVOLUTION_WIDTH = 3
MAX_LIGHTNESS_VALUE = 255
TRASHOLD_CONV_VALUE = 250
ADJACENT_BLOCKS_RATIO_TO_Y = 0.75
SHARED_LINE_TOLERANCE_RATIO = 0.5
OUT_OF_SPLITLINE_TOLERANCE = 0.13
VERTICALLY_ADJACENT_BLOCK_DISTANCE = 50
VERTICALLY_ADJACENT_BLOCK_VERTICAL_DISTANCE_RATIO = 0.9


@dataclass
class TableHeader:
    header_image: np.ndarray
    ocr_agent: lp.TesseractAgent
    header_content: ClassVar[list]
    splitlines: ClassVar[list]
    column_left_coordinates_list: ClassVar[list]

    def __post_init__(self):
        self.header_content = []
        self.splitlines = []
        self.column_left_coordinates_list = []

    def collect_header_elements(self):
        res = self.ocr_agent.detect(self.header_image, return_response=True)

        res['data'].apply(lambda row: get_header_elements(row, self.header_content),axis=1)
    
    def unite_adjacent_horizontal_blocks(self):
        """Соединяет блоки горизонтально""" 
        blocks_put_together = []
        visited_blocks = set()

        for block_num_i, block_i in enumerate(self.header_content):
            current_union_block = None
            for block_num_j, block_j in enumerate(self.header_content[block_num_i:], block_num_i):
                if block_num_i in visited_blocks:
                    break
                if current_union_block is None:
                    current_union_block = block_i

                if block_num_i != block_num_j or block_num_i == len(self.header_content) - 1:
                    if are_blocks_horizontally_adjacent(current_union_block,
                                                        block_j)\
                                            and block_num_j not in visited_blocks:
                        current_union_block = get_blocks_union(current_union_block,
                                                                block_j)
                        visited_blocks.add(block_num_j)

            if current_union_block is not None:
                blocks_put_together.append(current_union_block)    

        self.header_content = blocks_put_together
    
    def unite_adjacent_vertical_blocks(self):
        """Объединение многострочных выражений"""
        blocks_put_together = []
        visited_blocks = set()

        for block_num_i, block_i in enumerate(self.header_content):
            current_union_block = None
            for block_num_j, block_j in enumerate(self.header_content[block_num_i:], block_num_i):
                if block_num_i != block_num_j or block_num_i == len(self.header_content) - 1:
                    if current_union_block is None:
                        current_union_block = block_i
                    if are_blocks_vertically_adjacent(current_union_block,
                                                        block_j)\
                                                            and block_num_i not in visited_blocks:
                        if not are_devided_by_splitline_from_list(current_union_block,
                                                                    block_j,
                                                                    self.splitlines):                              
                            current_union_block = get_blocks_union(current_union_block,
                                                                    block_j,
                                                                    union_str='\n')
                            visited_blocks.add(block_num_j)

            if block_num_i not in visited_blocks:
                blocks_put_together.append(current_union_block)

        self.header_content = blocks_put_together

    def monoidex_structuring(self):
        """Структурирование по простым индексам"""
        table_header_list = []
        for block_num_i in range(len(self.header_content)):
            table_header_list.append(self.header_content[block_num_i]['text'])
            self.column_left_coordinates_list.append(self.header_content[block_num_i]['x_1'])

        self.header_content = table_header_list

    def multiindex_structuring(self):
        """Структурирование по мультииндексам"""
        table_header_list = []
        visited_nodes = set()

        for block_num_i, block_i in enumerate(self.header_content):
            for block_num_j, block_j in enumerate(self.header_content[block_num_i:], block_num_i):
                if block_num_i != block_num_j or block_num_i == len(self.header_content) - 1:
                    if are_devided_by_splitline_from_list(block_i,
                                                block_j,
                                                    self.splitlines)\
                                        and block_num_j not in visited_nodes:
                        table_header_list.append((block_i['text'],
                                                    block_j['text']))
                        visited_nodes.add(block_num_i)
                        visited_nodes.add(block_num_j)
                        self.column_left_coordinates_list.append(block_j['x_1'])
            if block_num_i not in visited_nodes:
                table_header_list.append((block_i['text'],''))
                visited_nodes.add(block_num_i)
                self.column_left_coordinates_list.append(block_i['x_1'])

        self.header_content = table_header_list

    def contains_multiindex(self):
        for block_num_i, block_i in enumerate(self.header_content):
            for block_num_j, block_j in enumerate(self.header_content[block_num_i:],block_num_i):
                if block_num_i != block_num_j or block_num_i == len(self.header_content) - 1:
                    if are_devided_by_splitline_from_list(block_i,
                                                        block_j,
                                                        self.splitlines):
                        return True
        return False

    def index_structuring(self):
        if self.contains_multiindex():
            self.multiindex_structuring()
        else:
            self.monoidex_structuring()

    def get_block_list(self, text_to_remove=' '):
        """Удаляем блок с определённым содержанием"""
        self.header_content = [block for block in self.header_content if block['text'] != text_to_remove]

        
def get_line_length(x_1,y_1,x_2,y_2):
    return np.sqrt((x_2 - x_1)**2 + (y_2 - y_1)**2)

# Получаем элемент хэддера таблицы
def get_header_elements(row, table_header=None):
    try:
        block_bottom = row['top'] + row['height']
        block_left_side = row['left'] + row['width']
        if str(row['text']) != 'nan':
            table_header.append({
                'text': row['text'],
                'y_1': row['top'],
                'y_2': block_bottom,
                'x_1': row['left'],
                'x_2': block_left_side
                })
    except TypeError:
        pass


# Получим высоту блока
def get_block_height(block):
    return block['y_2'] - block['y_1']

# Проверка на смежность блоков горизонтальных блоков
def are_blocks_horizontally_adjacent(block_a,block_b):
    block_a_height, block_b_height = get_block_height(block_a), get_block_height(block_b)
    max_block_height = max(block_a_height, block_b_height)
    are_x_adjacent = abs(block_b['x_1'] - block_a['x_2'])\
        < ADJACENT_BLOCKS_RATIO_TO_Y * max_block_height
    do_blocks_overlap = block_b['x_1'] < block_a['x_2'] and block_a['x_1'] < block_b['x_1']
    are_blocks_on_the_same_line = abs(block_b['y_2'] - block_a['y_2']) <\
        max_block_height * SHARED_LINE_TOLERANCE_RATIO
    return are_blocks_on_the_same_line and (are_x_adjacent or do_blocks_overlap)

def are_blocks_vertically_adjacent(block_a,block_b):
    return abs(block_b['x_1'] - block_a['x_1']) < VERTICALLY_ADJACENT_BLOCK_DISTANCE\
        and block_a['y_2'] * VERTICALLY_ADJACENT_BLOCK_VERTICAL_DISTANCE_RATIO <= block_b['y_1']

# Получаем координаты блока
def get_bbox_coordinates(block):
    return block['y_1'], block['y_2'], block['x_1'], block['x_2']

# Получаем объединённый bounding box
def get_bboxes_coordinates_union(block_a,block_b):
    y_1_a, y_2_a, x_1_a, x_2_a = get_bbox_coordinates(block_a)
    y_1_b, y_2_b, x_1_b, x_2_b = get_bbox_coordinates(block_b)

    y_1_union, x_1_union = min(y_1_a,y_1_b), min(x_1_a,x_1_b)
    y_2_union, x_2_union = max(y_2_a,y_2_b), max(x_2_a,x_2_b)

    return y_1_union, y_2_union,  x_1_union, x_2_union

# Получаем объединение текстовых блоков
def get_text_block_union(block_a,block_b,union_str):
    return block_a['text'] + union_str +  block_b['text']

# Получаем объединённый блок
def get_blocks_union(block_a, block_b, union_str = ' '):
    y_1_union, y_2_union,  x_1_union, x_2_union = get_bboxes_coordinates_union(block_a,block_b)
    return {
                'text': get_text_block_union(block_a,block_b,union_str),
                'y_1': y_1_union,
                'y_2': y_2_union,
                'x_1': x_1_union,
                'x_2': x_2_union
                }

# удаляем блок с определённым содержанием
def get_block_list(block_list,text_to_remove=' '):
    for block in block_list:
        if block['text'] == text_to_remove:
            block_list.remove(block)
    return block_list

def is_in_splitline_region_by_x(block,x_1_sl,y_1_sl,x_2_sl,y_2_sl):
    return block['x_1'] > x_1_sl * (1 - OUT_OF_SPLITLINE_TOLERANCE)\
        and block['x_2'] < x_2_sl * (1 + OUT_OF_SPLITLINE_TOLERANCE)

def is_above_the_splitline(block,x_1_sl,y_1_sl,x_2_sl,y_2_sl):
    return block['y_2'] <= y_1_sl * (1 + OUT_OF_SPLITLINE_TOLERANCE)

def are_devided_by_splitline(block_a,block_b,splitline):
    if is_in_splitline_region_by_x(block_a,*splitline) and\
        is_in_splitline_region_by_x(block_b,*splitline):
        if is_above_the_splitline(block_a,*splitline):
            return not is_above_the_splitline(block_b,*splitline)
    return False

def are_devided_by_splitline_from_list(block_a,block_b,splitlines):
    
    for splitline in splitlines:
        if splitlines is None:
            splitlines = []
        if are_devided_by_splitline(block_a,block_b,splitline):
            return True
    return False