from typing import Tuple, Mapping, List
import cv2 as cv
import numpy as np
import pygmo as pg
import matplotlib.pyplot as plt


def title(*args, **kwargs):
    print('=' * 30, *args, '='*30, **kwargs)


class ImageOptimizationProblem:

    def __init__(self, img_path: str=''):

        self.img = cv.imread(img_path)
        self.x_min = 0
        self.x_max = self.img.shape[0]
        self.y_min = 0
        self.y_max = self.img.shape[1]

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return ([self.x_min, self.y_min], [self.x_max, self.y_max])

    def fitness(self, x: np.ndarray) -> List[float]:
        function_value = sum(self.img[int(x[0]), int(x[1])])
        return [function_value]


# Инициализируем UDP (User Defined Problem)
udp = ImageOptimizationProblem('cat.jpg')

# Создадим объект pygmo
prob = pg.problem(udp)

# Информация задачи
title('Problem info')
print(prob)

# Количество поколений
number_of_generations = 1

# Фиксированный сид
current_seed = 171015

# Создание объекта дифференциальной эволюции, передав количество поколений в качестве входных данных
de_algo = pg.de(gen=number_of_generations, seed=current_seed)

# Создадим объект-алгоритм pygmo
algo = pg.algorithm(de_algo)

# Информация алгоритма
title('Algorithm info')
print(algo)

# Размер популяции
pop_size = 1000

# Создадим популяцию
pop = pg.population(prob, size=pop_size, seed=current_seed)

# Изучим популяцию (ДОЛГО)
inspect_pop = False
if inspect_pop:
    print(pop)


# Количество Эволюций
number_of_evolutions = 100

# Пустые контейнеры
individuals_list = []
fitness_list = []

# Произведём эволюцию несколько раз
for i in range(number_of_evolutions):
    pop = algo.evolve(pop)
    individuals_list.append(pop.get_x()[pop.best_idx()])
    fitness_list.append(pop.get_f()[pop.best_idx()])

# Выведем наилучшие образцы
title('Champion info')
print('Fitness (= function) value: ', pop.champion_f)
print('Decision variable vector: ', pop.champion_x)
print('Number of function evaluations: ', pop.problem.get_fevals())
print('Difference wrt the minimum: ', pop.champion_x - np.array([3,2]))

# Вытащим лучших индивидов из всех поколений
best_x = [ind[1] for ind in individuals_list]
best_y = [ind[0] for ind in individuals_list]

# Выразим границы задачи
(x_min, y_min), (x_max, y_max) = udp.get_bounds()

# Изобразим минимизацию за все поколения
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(np.arange(0, number_of_evolutions), fitness_list, label='Function values')

# Выведем лучший образец
champion_n = np.argmin(np.array(fitness_list))
ax.scatter(champion_n, np.min(fitness_list), marker='x', color='r', label='All time champion')

# Оформление
ax.set_xlim((0, number_of_evolutions))
ax.grid('major')
ax.set_title('The best representative of each epoch')
ax.set_xlabel('Epoch number')
ax.set_ylabel(r'Image pixels values $f(x,y)$')
ax.legend(loc='upper right')
ax.set_yscale('log')
plt.tight_layout()

# Показ
plt.savefig('best_image.png')
plt.clf()


image = cv.imread('cat.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

plt.imshow(image, cmap=plt.get_cmap('gray'))
plt.scatter(best_x,best_y, marker='x', color="violet")
ax.set_xlim((x_min, x_max))
ax.set_ylim((y_min, y_max))
ax.set_title('Функция на картинке', fontweight='bold')
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.tight_layout()

plt.savefig('image_result.png')
