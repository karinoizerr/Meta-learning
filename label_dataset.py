"""Скрипт для разметки данных."""

import pandas as pd
import os


TABLE_FILE_NAME = 'table.csv'


def get_winrate(algorithm, map_name):
    """
    Запускает алгоритм с необходимой картой.
    Алгоритм записывает свой винрейт в tmp.csv.
    """
    PyMARL_algorithms = [
        'qmix',
        'facmaddpg',
        'dop'
    ]

    if algorithm == 'q-learning':
        os.system('python smacexp20learn2.py'.format(map_name))
        os.system('python smacexp21test2.py'.format(map_name))
    elif algorithm in PyMARL_algorithms:
        os.system(
            'python src/main.py --config={} --env-config=sc2 with env_args.map_name={}'.format(algorithm, map_name))
    elif algorithm == 'dqn':
        os.system('python DQN.py'.format(map_name))


def get_best_algorithm(map_name):
    """Находит лучший алгоритм для карты."""
    algorithms = [
        'q-learning',
        'qmix',
        'facmaddpg',
        'dop',
        'dqn'
    ]

    df = pd.DataFrame(columns=algorithms)  # создали DataFrame с заголовкам
    for algorithm in algorithms:  # перебор по названиям алгоритмов
        df.to_csv('tmp.csv', index=False)  # перезаписали его в .csv файл
        get_winrate(algorithm, map_name)  # записываем винрейт алгоритма

    df = pd.read_csv('tmp.csv')  # снова читаем .csv файл с уже записанными винрейтами
    some_dict = df.to_dict()  # переводим DataFrame в словарь

    best_winrate = None
    best_algorithm = None
    for algorithm, winrate in some_dict.items():
        if best_winrate is None or winrate > best_winrate:
            best_winrate = winrate
            best_algorithm = algorithm
    return best_algorithm


def write_labels(filename):
    """
    Записывает в .csv файл метки.

    filename - название .csv файла.

    Читает .csv файл. Вытаскивает из него названия карт в список.
    По этому списку получает список названий лучших алгоритмов
    для каждой карты соответственно. Записывает список лучших
    алгоритмов как столбец с названием 'метка'.
    """
    df = pd.read_csv(filename)  # читаем .csv файлик в DataFrame
    map_names = df['название карты'].tolist()  # вытаскиваем названия карт из DataFrame в список
    labels = []  # создаём пустой список для меток
    for map_name in map_names:  # перебор по названиям карт
        labels.append(get_best_algorithm(map_name))  # добавляем название лучшего алгоритма для карты в список
    df['метка'] = labels  # добавляем в DataFrame столбец с метками
    df.to_csv(filename, index=False)  # записываем DataFrame обратно в .csv файл


def main():
    write_labels(TABLE_FILE_NAME)


if __name__ == '__main__':
    main()
