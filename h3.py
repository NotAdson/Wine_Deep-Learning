import pandas as pd
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

table = pd.read_csv("fuel.csv")

train_data = table.sample(frac=0.6, random_state=0)
valid_data = table.drop(train_data.index)

# Identifique colunas categóricas
categorical_cols = train_data.select_dtypes(include=['object']).columns

# Crie DataFrames com colunas categóricas transformadas em colunas binárias
train_data_dummies = pd.get_dummies(train_data, columns=categorical_cols)
valid_data_dummies = pd.get_dummies(valid_data, columns=categorical_cols)

# Use o MinMaxScaler para normalizar os dados mantendo as estatísticas do conjunto de treinamento
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data_dummies.drop("FE", axis=1))
valid_data_scaled = scaler.transform(valid_data_dummies.drop("FE", axis=1))

# Crie DataFrames normalizados
train_data_normalized = pd.DataFrame(train_data_scaled, columns=train_data_dummies.columns[:-1])
valid_data_normalized = pd.DataFrame(valid_data_scaled, columns=valid_data_dummies.columns[:-1])

# Separar características e rótulos
x_train = train_data_normalized
y_train = train_data["FE"]

x_valid = valid_data_normalized
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

history = model.fit(
    x_train, y_train,
    validation_data=(x_valid, y_valid),
    batch_size = 64,
    epochs = 200
)

history_df = pd.DataFrame(history.history)
history_df['loss'].plot()
plt.show()
