from block import *


class Region(Block):
    def __init__(self, x, y, data, size):
        super().__init__(x, y, data, 0)
        self.size = size
        self.find_domain = False
        self.best_domain = None

    def output_data(self, optimal_domain, contrast, brightness):
        self.find_domain = True
        #contrast, brightness = self.find_contrast_and_brightness2(optimal_domain)
        return [self.x, self.y, self.size, optimal_domain.code, optimal_domain.w, contrast, brightness]

    def find_contrast_and_brightness2(self, domain_data):
        # Fit the contrast and the brightness
        #domain_data = domain.reduce(domain.data, len(domain.data)//len(self.data))
        A = np.concatenate((np.ones((domain_data.size, 1)), np.reshape(domain_data, (domain_data.size, 1))), axis=1)
        b = np.reshape(self.data, (np.array(self.data).size,))
        x, _, _, _ = np.linalg.lstsq(A, b)
        contrast = x[1]
        brightness = x[0]

        # Ограничиваем яркость в пределах 0-255
        brightness = np.clip(brightness, 0, 255)
        # contrast = np.clip(contrast, 0, 1)

        return contrast, brightness
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
