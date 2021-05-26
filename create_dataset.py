"""Файл для создания датасета."""

import pandas as pd


TABLE_FILE_NAME = 'table.csv'
COLUMNS = [
    'название карты',
    'количество дружественных агентов',
    'количество вражеских агентов',
    'координата x дружественных агентов',
    'координата y дружественных агентов',
    'координата x вражеских агентов',
    'координата y вражеских агентов',
    'наличие препятствия',
    'координата x начала стены',
    'координата y начала стены',
    'координата x середины стены',
    'координата y середины стены',
    'координата x конца стены',
    'координата y конца стены'
]


def add_new_point_to_csv(filename, map_name, n_friends, n_enemies, friends_coord, enemies_coord, wall_exists, wall_begins, wall_middle, wall_ends):
    """Записывает точку данных в табличку, которая находится в .csv файлике."""
    x_friends_coord, y_friends_coord = friends_coord
    x_enemies_coord, y_enemies_coord = enemies_coord
    x_wall_begins, y_wall_begins = wall_begins
    x_wall_middle, y_wall_middle = wall_middle
    x_wall_ends, y_wall_ends = wall_ends
    new_row = {
        'название карты': map_name,
        'количество дружественных агентов': n_friends,
        'количество вражеских агентов': n_enemies,
        'координата x дружественных агентов': x_friends_coord,
        'координата y дружественных агентов': y_friends_coord,
        'координата x вражеских агентов': x_enemies_coord,
        'координата y вражеских агентов': y_enemies_coord,
        'наличие препятствия': wall_exists,
        'координата x начала стены': x_wall_begins,
        'координата y начала стены': y_wall_begins,
        'координата x середины стены': x_wall_middle,
        'координата y середины стены': y_wall_middle,
        'координата x конца стены': x_wall_ends,
        'координата y конца стены': y_wall_ends
    }
    df = pd.read_csv(filename)
    df = df.append(new_row, ignore_index=True)
    df.to_csv(filename, index=False)


def main():
    table = pd.DataFrame(columns=COLUMNS)  # создаём табличку только с заголовками
    table.to_csv(TABLE_FILE_NAME, index=False)  # сохраняем эту табличку в .csv файл

    # записываем экземпляры точек данных в табличку
    #add_new_point_to_csv(TABLE_FILE_NAME, '2m2mFOX', 2, 2, (1, 1), (10, 1), 0, (2, 2), (4, 2), (8, 2))


if __name__ == '__main__':
    main()
