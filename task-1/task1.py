import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Импортируем данные
data = pd.read_csv('Features_Variant_1.csv', sep=',')

# Проверим, что кол-во null-ов равно 0 у всех фич
# print(f'Number of null elements in the dataset:\n{data.isnull().sum()}')

# Проверим корреляцию между элементами
# Построим тепловую карту для более легкого определения корреляции между переменными (см. в Jupyter)
# plt.subplots(figsize=(10,10))
# sns.heatmap(data.corr())

# Выберем ТОП фич на основе знания о корреляции
# feature_list = data.corr()["output"].sort_values()
# print(feature_list)

# Выбранные финальные фичи
Features_final = ['baseTime', 'Unnamed: 24', 'Unnamed: 22', 'Unnamed: 7', 'commBase', 'Unnamed: 21', 'Unnamed: 6', 'comm24_1', 'Unnamed: 11', 'diff_24,48', 'comm24', 'output']
data = data[Features_final]

# Разбиваем данные на X и Y
X_data = data.drop(columns=['output'])
Y_data = data['output']

X_data_np = X_data.to_numpy()
Y_data_np = Y_data.to_numpy()

# Стандартизируем данные
scaler = MinMaxScaler()
X_data_np = scaler.fit_transform(X_data_np)

# Root Mean Squared Error
def rmse(fact: np.ndarray, predicted: np.ndarray):
    return np.sqrt(np.mean((fact - predicted) ** 2))

# Коэффициент детерминизации
def r2(fact: np.ndarray, predicted: np.ndarray):
    return 1 - (np.sum((fact - predicted) ** 2) / np.sum((fact - np.mean(fact)) ** 2))

# Градиентный спуск
def gradient_descent(X: np.ndarray, Y: np.ndarray):
    n, m = X.shape
    weights = np.ones(m)
    lamb = 0.15

    for i in range(1,15000):
        prediction = X @ weights
        X_trasnposed = X.transpose()
        grad = (X_trasnposed @ (prediction - Y)) / n
        weights = weights - lamb * grad
        if np.linalg.norm(grad) < 0.1:
            return weights

    return weights

# Cross-Validation

# Разбиваем на 5 фолдов
folds_X = np.array_split(X_data_np, 5)
folds_Y = np.array_split(Y_data_np, 5)

# Тренируемся
cv_results_train = []
cv_results_test = []

for i in range(5):
    # Будем тренировать i-ую модель на всех фолдах кроме i-го
    current_X_train = np.vstack([folds_X[j] for j in range(5) if j != i])
    current_Y_train = np.hstack([folds_Y[j] for j in range(5) if j != i])

    # Будем тестировать i-ую модель как-раз на i-ом фолде
    current_X_test = folds_X[i]
    current_Y_test = folds_Y[i]

    # Поехали - считаем веса
    weights = gradient_descent(current_X_train, current_Y_train)

    # Считаем RMSE и R^2 для Train
    current_rmse_train = rmse(current_Y_train, current_X_train @ weights)
    current_r2_train = r2(current_Y_train, current_X_train @ weights)
    cv_results_train.append((current_rmse_train, current_r2_train))

    # Считаем RMSE и R^2 для Test
    current_rmse_test = rmse(current_Y_test, current_X_test @ weights)
    current_r2_test = r2(current_Y_test, current_X_test @ weights)
    cv_results_test.append((current_rmse_test, current_r2_test))

    print(f'\nFold {i+1}\n----')
    print(f'RMSE-train: {current_rmse_train:.3f}')
    print(f'R^2-train: {current_r2_train:.3f}\n')

    print(f'RMSE-test: {current_rmse_test:.3f}')
    print(f'R^2-test: {current_r2_test:.3f}\n')

# Возьмем среднее и получим оценку обобщающей способности нашего алгоритма
print(f'STD RMSE-train: {np.std([rmse_err for rmse_err, _ in cv_results_train]):.3f}')
print(f'STD R^2-train: {np.std([r2_err for _, r2_err in cv_results_train]):.3f}')
print(f'Mean RMSE-train: {np.mean([rmse_err for rmse_err, _ in cv_results_train]):.3f}')
print(f'Mean R^2-train: {np.mean([r2_err for _, r2_err in cv_results_train]):.3f}\n')

print(f'STD RMSE-test: {np.std([rmse_err for rmse_err, _ in cv_results_test]):.3f}')
print(f'STD R^2-test: {np.std([r2_err for _, r2_err in cv_results_test]):.3f}')
print(f'Mean RMSE-test: {np.mean([rmse_err for rmse_err, _ in cv_results_test]):.3f}')
print(f'Mean R^2-test: {np.mean([r2_err for _, r2_err in cv_results_test]):.3f}\n')
