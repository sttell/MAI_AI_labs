"""Модуль для работы с прямыми.
"""
import numpy as np
from typing import Tuple


class Line:
    def __init__(self, points: np.ndarray) -> None:
        self.k = None
        self.b = None
        self.points = points

    def estimate_params(self) -> None:
        """
        Оценка параметров прямой по точкам.
        В текущей реализации возможна оценка только по двум точкам
        """
        if len(self.points) > 2:
            raise NotImplementedError
        elif len(self.points) < 2:
            raise ValueError(f"Not enough points. Must be at least 2, but got {len(self.points)}")

        x1, y1 = self.points[0]
        x2, y2 = self.points[1]

        self.k = (y1 - y2) / (x1 - x2 + 0.000001)
        self.b = y2 - self.k * x2

    def divide_points(self, points: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Разделение точек на принадлежащие к прямой и не принадлежащие.
        Требует вызова estimate_params перед использованием

        Args:
            points: np.ndarray - Массив точек в формате [[x1, y1], ..., [xn, yn]]
            eps: float - Расстояние от прямой внутри которого будет принято решение о принадлежности точки прямой.
        """

        if self.k is None or self.b is None:
            raise RuntimeError("You must call the Line::estimate_params method before divide points.")

        distance = np.abs(self.k * points[:, 0] - points[:, 1] + self.b) / np.sqrt(self.k ** 2 + 1 + 0.00001)
        return points[distance <= eps], points[distance > eps]
