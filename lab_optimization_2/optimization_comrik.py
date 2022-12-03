from typing import Tuple, Mapping, List

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
import numpy as np
from numpy import random

import pygmo as pg
import math


def title(*args, **kwargs):
    print('=' * 30, *args, '='*30, **kwargs)


class McComrickOptimizationProblem:

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        return ([self.x_min, self.y_min], [self.x_max, self.y_max])

    def fitness(self, x: np.ndarray) -> List[float]:
        function_value = math.sin(x[0] + x[1]) + math.pow(x[0] - x[1], 2.0) - 1.5*x[0] + 2.5*x[1]+1 +2 #чтобы график работал
        return [function_value]


# Инициализируем UDP (User Defined Problem)
udp = McComrickOptimizationProblem(-1.5, 4.0, -3.0, 4.0)

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
best_x = [ind[0] for ind in individuals_list]
best_y = [ind[1] for ind in individuals_list]

# Выразим границы задачи
(x_min, y_min), (x_max, y_max) = udp.get_bounds()

# Изобразим минимизацию за все поколения
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(np.arange(0, number_of_evolutions), fitness_list, label='Function value')

# Выведем лучший образец
champion_n = np.argmin(np.array(fitness_list))
ax.scatter(champion_n, np.min(fitness_list), marker='x', color='r', label='Champion')

# Оформление
ax.set_xlim((0, number_of_evolutions))
ax.grid('major')
ax.set_title('The best representative of each epoch')
ax.set_xlabel('Epoch number')
ax.set_ylabel(r'McComrick function value $f(x,y)$')
ax.legend(loc='upper right')
ax.set_yscale('log')
plt.tight_layout()

# Показ
plt.savefig('best_mccomrick.png')
plt.clf()


# Изобразим функцию МакКомрика
grid_points = 100
x_vector = np.linspace(x_min, x_max, grid_points)
y_vector = np.linspace(y_min, y_max, grid_points)
x_grid, y_grid = np.meshgrid(x_vector, y_vector)
z_grid = np.zeros((grid_points, grid_points))
for i in range(x_grid.shape[1]):
    for j in range(x_grid.shape[0]):
        z_grid[i, j] = udp.fitness([x_grid[i, j], y_grid[i, j]])[0]

# Create figure
fig, ax = plt.subplots(figsize=(9,5))
cs = ax.contour(x_grid, y_grid, z_grid, 50)

# Показываем лучших представителей за каждое поколение
ax.scatter(best_x, best_y, marker='x', color='r')

# Оформление
ax.set_xlim((x_min, x_max))
ax.set_ylim((y_min, y_max))
ax.set_title('McComrick function')
ax.set_xlabel('X')
ax.set_ylabel('Y')
cbar = fig.colorbar(cs)
cbar.ax.set_ylabel(r'McComrick function value $f(x,y)$')
plt.tight_layout()

# Показ
plt.savefig('mccomrick_result.png')
plt.clf()