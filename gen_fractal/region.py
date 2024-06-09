from domain import *


class Region(Domain):
    def __init__(self, x, y, data, size):
        super().__init__(x, y, data, 0)
        self.size = size
        self.code = None

    def output_data(self, optimal_domain):
        return [self.x, self.y, self.size, optimal_domain.code, optimal_domain.w, optimal_domain.contrast, optimal_domain.brightness]

    def decompose_region(self):
        """
        Функция для деления рангового блока на 4 части с заданным размером региона.

        Аргументы:
        - I: numpy.ndarray, исходный ранговый блок
        - region_size: int, размер региона

        Возвращает:
        - regions: list, список содержащий 4 региона

        """
        new_size = self.size//2
        # Получаем размеры исходного блока
        block_height, block_width = self.size, self.size

        # Размер каждой части
        part_height = block_height // 2
        part_width = block_width // 2

        data = self.data
        # Разбиваем ранговый блок на 4 части
        region1_data = data[:part_height, :part_width]
        region2_data = data[:part_height, part_width:]
        region3_data = data[part_height:, :part_width]
        region4_data = data[part_height:, part_width:]

        x, y = self.x, self.y
        region1 = Region(x=x, y=y, data=region1_data, size = new_size)
        region2 = Region(x=x+part_width, y=y, data=region2_data, size=new_size)
        region3 = Region(x=x, y=y+part_height, data=region3_data, size=new_size)
        region4 = Region(x=x+part_width, y=y+part_height, data=region4_data, size=new_size)

        return [region1, region2, region3, region4]
