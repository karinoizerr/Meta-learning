import pandas as pd


def add_new_row_to_csv(filename, n_friends, n_enemies, friends_coord, enemies_coord, wall_exists, wall_begins, wall_middle, wall_ends):
    x_friends_coord, y_friends_coord = friends_coord
    x_enemies_coord, y_enemies_coord = enemies_coord
    x_wall_begins, y_wall_begins = wall_begins
    x_wall_middle, y_wall_middle = wall_middle
    x_wall_ends, y_wall_ends = wall_ends
    new_row = {
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
    df.append(new_row)
