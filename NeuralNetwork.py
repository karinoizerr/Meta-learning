import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import pandas as pd
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras


TABLE_FILE_NAME = 'table.csv'
NUM_FEATURES = (13, )
NUM_CLASSES = 5
LAYER_OUTPUT_WIDTH = 100

dataframe = pd.read_csv(TABLE_FILE_NAME)  # читаем .csv файл в DataFrame
dataframe.head()
def label_to_code(label):
    result = None
    if label == 'q-learning':
        result = 0
    elif label == 'qmix':
        result = 1
    elif label == 'facmaddpg':
        result = 2
    elif label == 'dop':
        result = 3
    elif label == 'dqn':
        result = 4
    return result

dataframe['метка'] = dataframe['метка'].apply(label_to_code)
dataframe

DATASET_SIZE, _ = dataframe.shape
val_size = int(0.1 * DATASET_SIZE)
test_size = int(0.1 * DATASET_SIZE)
train_size = DATASET_SIZE - val_size - test_size

print('Обучающих точек данных: {}'.format(train_size))
print('Валидационных точек данных: {}'.format(val_size))
print('Тестовых точек данных: {}'.format(test_size))


def dataframe_to_dataset(dataframe):
    """Читает из датафрейма только точки данных и переводит в dataset."""
    dataframe = dataframe.copy()  # копируем входной DataFrame
    dataframe.pop('название карты')  # отбрасываем столбец с названиями карт
    labels = dataframe.pop('метка')  # вытаскиваем из DataFrame метки
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))  # создаём dataset
    return dataset


full_dataset = dataframe_to_dataset(dataframe)  # из DataFrame создаём Dataset
shuffled_dataset = full_dataset.shuffle(buffer_size=DATASET_SIZE)  # перемешиваем
val_dataset = shuffled_dataset.take(val_size)  # выдёргиваем из всего датасета валидационную часть
tmp_dataset = shuffled_dataset.skip(val_size)  # оставляем часть от целого без валидационной
test_dataset = tmp_dataset.take(test_size)  # выдёргиваем из оставшейся части тестовую
train_dataset = tmp_dataset.skip(test_size)  # в обучающей части остаётся остальное

for x, y in train_dataset.take(1):
    print("Input:", x)
    print("Target:", y)

train_dataset = train_dataset.batch(1)
val_dataset = val_dataset.batch(1)
test_dataset = test_dataset.batch(1)



def encode_numerical_feature(feature, name, dataset):
    """."""
    normalizer = Normalization()  # Create a Normalization layer for our feature

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    normalizer.adapt(feature_ds)  # Learn the statistics of the data

    encoded_feature = normalizer(feature)  # Normalize the input feature
    return encoded_feature


def encode_categorical_feature(feature, name, dataset):
    """."""
    lookup = IntegerLookup(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    lookup.adapt(feature_ds)  # Learn the set of possible string values and assign them a fixed integer index

    encoded_feature = lookup(feature)  # Turn the string input into integer indices
    return encoded_feature


n_friends = keras.Input(shape=(1,), name="количество дружественных агентов")
n_enemies = keras.Input(shape=(1,), name="количество вражеских агентов")
x_friends_coord = keras.Input(shape=(1,), name="координата x дружественных агентов")
y_friends_coord = keras.Input(shape=(1,), name="координата y дружественных агентов")
x_enemies_coord = keras.Input(shape=(1,), name="координата x вражеских агентов")
y_enemies_coord = keras.Input(shape=(1,), name="координата y вражеских агентов")
wall_exist = keras.Input(shape=(1,), name="наличие препятствия", dtype="int64")
x_wall_begins = keras.Input(shape=(1,), name="координата x начала стены")
y_wall_begins = keras.Input(shape=(1,), name="координата y начала стены")
x_wall_middle = keras.Input(shape=(1,), name="координата x середины стены")
y_wall_middle = keras.Input(shape=(1,), name="координата y середины стены")
x_wall_ends = keras.Input(shape=(1,), name="координата x конца стены")
y_wall_ends = keras.Input(shape=(1,), name="координата y конца стены")

all_inputs = [
    n_friends,
    n_enemies,
    x_friends_coord,
    y_friends_coord,
    x_enemies_coord,
    y_enemies_coord,
    wall_exist,
    x_wall_begins,
    y_wall_begins,
    x_wall_middle,
    y_wall_middle,
    x_wall_ends,
    y_wall_ends
]

n_friends_encoded = encode_numerical_feature(n_friends, "количество дружественных агентов", train_dataset)
n_enemies_encoded = encode_numerical_feature(n_enemies, "количество вражеских агентов", train_dataset)
x_friends_coord_encoded = encode_numerical_feature(x_friends_coord, "координата x дружественных агентов", train_dataset)
y_friends_coord_encoded = encode_numerical_feature(y_friends_coord, "координата y дружественных агентов", train_dataset)
x_enemies_coord_encoded = encode_numerical_feature(x_enemies_coord, "координата x вражеских агентов", train_dataset)
y_enemies_coord_encoded = encode_numerical_feature(y_enemies_coord, "координата y вражеских агентов", train_dataset)
wall_exist_encoded = encode_categorical_feature(wall_exist, "наличие препятствия", train_dataset)
x_wall_begins_encoded = encode_numerical_feature(x_wall_begins, "координата x начала стены", train_dataset)
y_wall_begins_encoded = encode_numerical_feature(y_wall_begins, "координата y начала стены", train_dataset)
x_wall_middle_encoded = encode_numerical_feature(x_wall_middle, "координата x середины стены", train_dataset)
y_wall_middle_encoded = encode_numerical_feature(y_wall_middle, "координата y середины стены", train_dataset)
x_wall_ends_encoded = encode_numerical_feature(x_wall_ends, "координата x конца стены", train_dataset)
y_wall_ends_encoded = encode_numerical_feature(y_wall_ends, "координата y конца стены", train_dataset)

all_features = layers.concatenate(
    [
        n_friends_encoded,
        n_enemies_encoded,
        x_friends_coord_encoded,
        y_friends_coord_encoded,
        x_enemies_coord_encoded,
        y_enemies_coord_encoded,
        wall_exist_encoded,
        x_wall_begins_encoded,
        y_wall_begins_encoded,
        x_wall_middle_encoded,
        y_wall_middle_encoded,
        x_wall_ends_encoded,
        y_wall_ends_encoded
    ]
)

x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, rankdir="TB", dpi=100)

model.fit(train_dataset, epochs=500, validation_data=val_dataset)
model.evaluate(test_dataset)


"""
sample = {
    'количество дружественных агентов': 4,
    'количество вражеских агентов': 4,
    'координата x дружественных агентов': 3,
    'координата y дружественных агентов': 3,
    'координата x вражеских агентов': 15,
    'координата y вражеских агентов': 15,
    'наличие препятствия': 1,
    'координата x начала стены': 1,
    'координата y начала стены': 1,
    'координата x середины стены': 3,
    'координата y середины стены': 2,
    'координата x конца стены': 5,
    'координата y конца стены': 5
}
input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}


predict = model.predict(input_dict)


def predict_to_label(predict):
    label = None
    if int(predict[0][0]) == 0:
        label = 'q-learning'
    elif int(predict[0][0]) == 1:
        label = 'qmix'
    elif int(predict[0][0]) == 2:
        label = 'facmaddpg'
    elif int(predict[0][0]) == 3:
        label = 'dop'
    elif int(predict[0][0]) == 4:
        label = 'dqn'
    return label

print(predict_to_label(predict)) """