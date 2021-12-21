import layoutparser as lp
from dataclasses import dataclass
from modules.header_split_lines import get_filtered_table_header_lines
from modules.header_extractor import TableHeader, get_header_elements
from modules.split_table import split_table
from modules.table_components_to_json import process_table
from modules.table_parsing_by_column import TableBody


@dataclass
class TableStructureExtractor():
    image_path: str
    ocr_agent = lp.TesseractAgent(languages='eng')

    def __post_init__(self):
        self.header_image, self.body_image = split_table(self.image_path)
        self.header = TableHeader(self.header_image, self.ocr_agent)
        self.body = TableBody(self.body_image, self.ocr_agent)
        # self.body = TableBody(self.body_image)

    def get_header_list_of_index(self):
        """Возвращает список столбцов таблицы"""
        # Распознавание шапки таблицы и выбор информативных блоков
        self.header.collect_header_elements()
        # Выделение разделительных линий внутри шапки таблицы
        self.header.splitlines = get_filtered_table_header_lines(self.header_image)
        # Объединяем смежные горизонтальные блоки
        self.header.unite_adjacent_horizontal_blocks()
        # Избавляеся от пустых блоков
        self.header.get_block_list(text_to_remove=' ')
        # Объединяем смежные вертикальные блоки
        self.header.unite_adjacent_vertical_blocks()
        # Структурирование по индексу (сложный индекс и моноиндекс)
        self.header.index_structuring()

        return self.header.header_content

    def get_table_body_content(self):
        """Возвращает содержание тела таблицы"""
        # Координаты хэддера доступны после вызова метода get_header_list_of_index()
        column_coord = self.header.column_left_coordinates_list

        self.body.img_processing(column_coord)

        return self.body.final

    def get_table_content(self):
        """Возвращает таблицу pandas.DataFrame"""
        head_list = self.get_header_list_of_index()
        body_df = self.get_table_body_content()
        resulting_table = process_table(body_df, head_list)

        return resulting_table