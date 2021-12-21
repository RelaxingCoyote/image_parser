import layoutparser as lp
from dataclasses import dataclass
from modules.split_table import split_table

import pandas as pd
import numpy as np

DataFrame = pd.core.frame.DataFrame

ocr_agent = lp.TesseractAgent(languages='eng')

path = 'tables_to_test/image_666.jpg'

column_coord = [21, 152, 340, 514, 688, 791, 934, 1070]


def alignment_table(row, list_of_coordinates: list, by_witch_column: str) -> None:
    '''
    Function for detrmining whether an element belongs to a row or column
    '''

    for scope in list_of_coordinates:
        if row[by_witch_column] in range(scope[0], scope[1]):
            num = list_of_coordinates.index(scope) + 1

            return num

    return None

@dataclass
class TableBody:
    body: np.ndarray
    # column_coord: list
    ocr_agent: lp.TesseractAgent

    def get_np_array(self, body) -> None:
        '''
        function for turning an numpy.ndarray into a dictionary
        '''

        self.res = ocr_agent.detect(body, return_response=True)


    def df_preparation(self) -> None:
        '''
        function for determing and preprocessing df
        '''
        self.nan_value = float("NaN")
        self.df = pd.DataFrame(self.res['data'])

        # выделяем нужные столбцы и сортируем слова в нуджном порядке
        # убираем пробелы
        self.df = self.df[['left', 'top', 'text', 'width']].dropna().sort_values(by=['top', 'left']) 
        self.df['text'] = self.df['text'].str.lstrip()                                               
        self.df.replace("", self.nan_value, inplace=True)
        self.df.dropna(subset = ["text"], inplace=True)
        # Окей, сначала создаём наны, потом удаляем их
        # Нужно ли сортировать? Вроде, с порядком там всё было ок

        # возьмем сами слова как индексы, поскольку по ним удобнее ориентироваться
        # будем ориентироваться по концу слова
        self.df = self.df.set_index('text')                                                          
        self.df['left'] = self.df['left'] + self.df['width']
        # left = right. Hm...                                         

    # надо разбить на две функции, одну для столбцов, другу - для строк
    def separation_by_coordinates(self, coord: list, d: int, column: bool) -> None:
        '''
        For for seperating values by columns or rows it belongs to 
        '''
        self.edges = coord
        self.edges = [abs(x - d) for x in self.edges]
        self.edges.append(self.edges[-1] * 2)
        lst_of_ranges = [(i, j) for i, j in zip(self.edges[:-1], self.edges[1:])]

        if column:
            self.df['column'] = self.df.apply(
                lambda row: alignment_table(
                            row, 
                            list_of_coordinates = lst_of_ranges, 
                            by_witch_column = 'left'), 
                            axis=1)

        else:
            self.df['row'] = self.df.apply(
                lambda row: alignment_table(
                            row, 
                            list_of_coordinates = lst_of_ranges, 
                            by_witch_column = 'top'), 
                            axis=1) 
        

    def get_columns(self, column_coord: list) -> None:
        '''
        There we adding a column with column numbers to which the element belongs
        '''

        self.separation_by_coordinates(coord=column_coord, d=5, column=True)


    def get_rows(self, coord: list) -> None:
        '''
        There we adding a column with column numbers to which the element belongs
        '''
        self.separation_by_coordinates(coord=coord, d=10, column=False)


    def get_pivot_table(self) -> DataFrame:
        '''
        There we modify the df into convinient format
        '''

        self.df = self.df.reset_index()
        final = self.df.pivot_table(
                index=['row'],
                values=['text'],
                columns=['column'], 
                aggfunc=lambda x: ' '.join(list(dict.fromkeys(x.tolist()))))

        return final


    def img_processing(self, column_coord) -> DataFrame:
        
        self.path = path
        # self.get_body(self.path)
        self.get_np_array(self.body)
        self.df_preparation()
        self.get_columns(column_coord)
        self.row_coord = list(self.df[self.df['column'] == 1]['top'])
        self.get_rows(self.row_coord)
        self.final = self.get_pivot_table()
        # print(self.final) 


if __name__ == "__main__":
    header, body = split_table(path)
    # body_exctractor = TableBody(body, column_coord)
    body_exctractor = TableBody(body)
    body_exctractor.img_processing(column_coord)
    #img_processing(path)


