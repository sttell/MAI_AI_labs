'''
RANSAC for 2d lines

Algo:
    I Hypotesys generation stage
    1. Sample 2d ponts (1. 2 points; 2. 5 points)
    2. Model estimation (1. analytics, 2. MSE estimation)

    II Hypotesys evaluation stage
    3. Inlier counting (%inlinear > thresh)
        If True -> best params
        If False -> 1.
    4. # iter > num_iter?
'''

from typing import Dict
from line import Line
import numpy as np
import matplotlib.pyplot as plt


class RANSAC:
    """RANSAC algo class
    """
    def __init__(self) -> None:
        self.iter_num: int = 100
        self.inlin_thrsh: float = 0.8
        self.epsilon: float = 0.1
        self.best_params: dict = {}
        self.inliers:  list = []
        self.outliers: list = []
        self.score: int = 0
        self.points: np.ndarray = None

    def set_case(self,
                 points: np.ndarray,
                 iter_num: int = 100,
                 inline_threshold: float = 0.8,
                 epsilon: float = 0.1) -> None:
        """
        Устанавливает массив точек, проверяет что нужная размерность

        Args:
            points: np.ndarray - Обучающая выборка. Массив точек в формате [[x1, y1], ..., [xn, yn]]
            iter_num: int - Максимальное количество итераций. Положительное целое число. Defaults to 100.
            inline_threshold: float - Порог доли внутренних точек при котором отанавливается поиск оптимальных
                                      параметров. Положительное число в диапазоне [0.0, 1.0]. Defaults to 0.8
            epsilon: float - Величина расстояния внутри которого точки расположенные около оцениваемой линии считаются
                             принадлежащими этой линии. Положительное число. Defaults to 0.1
        """

        if len(points) < 2:
            raise ValueError("Points array must contain greater than two points.")
        if iter_num <= 0:
            raise ValueError("Number of iterations must be positive.")
        if inline_threshold < 0 or inline_threshold > 1:
            raise ValueError("Inliers threshold must be in range [0, 1]")
        if epsilon < 0.0:
            raise ValueError("Epsilon value must be positive.")

        self.points = points
        self.iter_num = iter_num
        self.inlin_thrsh = inline_threshold
        self.epsilon = epsilon

    def clear_case(self) -> None:
        """
        Чистит все параметры
        """
        self.best_params = {}
        self.inliers = []
        self.outliers = []
        self.points = []
        self.score = 0

    def fit(self) -> Dict[str, float]:
        """
        Производит оценку параметров прямой по заданному в set_case набору точек и параметрам алгоритма.

        Returns:
            Dict[str, float] - параметры прямой
                k: float - коэффициент наклона прямой
                b: float - коэффициент сдвига прямой
        """
        for i in range(self.iter_num):

            # Получаем 2 случайных точки из набора данных
            rnd_points_idx = np.random.randint(0, len(self.points), 2)
            point1 = self.points[rnd_points_idx[0]]
            point2 = self.points[rnd_points_idx[1]]

            # Оцениваем параметры линии по двум точкам
            line = Line(np.array([point1, point2]))
            line.estimate_params()

            # Разделяем набор данных на внутренние и внешние точки
            inliers, outliers = line.divide_points(self.points, self.epsilon)

            # Подсчитываем долю кол-ва внешних точек
            curr_outliers_ratio = len(outliers) / len(self.points)

            # В случае если итерация не первая
            if not (len(self.inliers) == 0 and len(self.outliers) == 0):

                # Подсчитываем долю кол-ва внешних точек для лучшего набора параметров
                best_outliers_ratio = len(self.outliers) / len(self.points)

                # Если результат не улучшился по отношению к лучшему найденному, то переходим на следующую итерацию
                if best_outliers_ratio < curr_outliers_ratio:
                    continue

            # В случае если результат улучшен записываем новые лучшие параметры
            self.best_params = {
                "k": line.k,
                "b": line.b
            }

            self.inliers = inliers
            self.outliers = outliers

            # Проверяем соответствуют ли критерию остановки текущие параметры
            if (1 - curr_outliers_ratio) > self.inlin_thrsh:
                break

        return self.best_params

    def draw(self, save_path: str) -> None:
        """
        Отрисовывает оцененный case на графике

        Args:
            save_path: str - Путь для сохранения графика
        """
        plt.figure(figsize=(15, 10))
        plt.scatter(self.inliers[:, 0], self.inliers[:, 1], c="blue", label="Inliers")
        plt.scatter(self.outliers[:, 0], self.outliers[:, 1], c="red", label="Outliers")

        xmin = min(self.inliers[:, 0].min(), self.outliers[:, 0].min())
        xmax = max(self.inliers[:, 0].max(), self.outliers[:, 0].max())
        line_x = np.linspace(xmin, xmax, 2)
        line_y = self.best_params['k'] * line_x + self.best_params['b']
        plt.plot(line_x, line_y, c='green', label='Estimated line')

        plt.grid()
        plt.legend()
        plt.title("RANSAC estimation example")
        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")

        plt.savefig(save_path, dpi=300)
