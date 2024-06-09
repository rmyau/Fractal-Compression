import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np
import random
from domain import *
# Parameters






# Define parameters
region_size = 16

pop_size = 100
num_generations = 10
crossover_rate = 0.8
mutation_rate = 0.2
d = Domain(x=5,y=3,data=[])
















#
# def generate_all_domain_blocks(img, source_size, step, source_blocks):
#     for k in range((img.shape[0] - source_size) // step + 1):
#         for l in range((img.shape[1] - source_size) // step + 1):
#             source_blocks.append(
#                 domain(data=img[k * step:k * step + source_size, l * step:l * step + source_size], x=l, y=k))
#
#
