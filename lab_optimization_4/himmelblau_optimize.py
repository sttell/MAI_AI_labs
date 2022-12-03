import optuna
import numpy as np

def Himmelblau(x: np.ndarray) -> np.float64:
    '''
    Функция Химмельблау
    
    Args:
        x(np.ndarray): Вектор аргументов
        
    Returns:
        np.float64: Результат функции
    '''
    return np.square(np.square(x[0]) + x[1] - 11.0) + np.square(x[0] + np.square(x[1]) - 7.0)


def objective(trial):
    x = np.array([0.0, 0.0], dtype=np.float64)
    x[0] = trial.suggest_float("x", -10, 10)
    x[1] = trial.suggest_float("y", -10, 10)
    return Himmelblau(x)

study = optuna.create_study()
study.optimize(objective, n_trials=100)

best_params = study.best_params
found_x = best_params["x"]
found_y = best_params["y"]

print('Analytic optimums:')
print('\tf(3, 2) = 0')
print('\tf(-2.805118, 3.131312) = 0')
print('\tf(-3.779310, -3.283186) = 0')
print('\tf(3.584428, -1.848126) = 0')
print(f"\nFound x: {found_x}, Found y: {found_y}, Result: {Himmelblau([found_x, found_y])}")