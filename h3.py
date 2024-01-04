import pandas as pd
from tensorflow.keras import layers, Sequential

table = pd.read_csv("fuel.csv")

train_data = table.sample(frac=0.6, random_state=0)
valid_data = table.drop(train_data.index)

list_max = train_data.max()
list_min = train_data.min()

train_data = (train_data - list_min) / (list_max - list_min)
valid_data = (valid_data - list_min) / (list_max - list_min)

x_train = train_data.drop("FE", axis=1)
x_valid = valid_data.drop("FE", axis=1)
y_train = valid_data["FE"]
y_valid = valid_data["FE"]

model = Sequential([
    layers.Dense(128, activation="relu", input_shape=[50]),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer = "adam",
    loss = "mae"
)

history = model.fit()


