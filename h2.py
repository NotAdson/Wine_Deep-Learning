import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#Ler a tabela
data = pd.read_csv("red-wine.csv")

#Separar os dados que serão utilizados para treinamento e validação
train_data = data.sample(frac=0.7, random_state=0)#Utiliza apenas 70% dos dados
valid_data = data.drop(train_data.index)


#Deixar os valores da tabela na mesma escala, para que nenum valor seja mais importante que outro.
list_max = train_data.max(axis=0)
list_min = train_data.min(axis=0)

train_data = (train_data - list_min) / (list_max - list_min)
valid_data = (valid_data - list_min) / (list_max - list_min)

#Separar data alvo
x_train = train_data.drop("quality", axis=1)
x_valid = valid_data.drop("quality", axis=1)
y_train = train_data["quality"]
y_valid = valid_data["quality"]

#Criar modelo
model = keras.Sequential([
    layers.Dense(512, activation="relu", input_shape=[11]),
    layers.Dense(512, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(1)
])

#Decidir arquitetura
model.compile(
    optimizer="adam",
    loss="mae"
)

#Botar modelo para rodar e salvar histórico
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=256, epochs=10)

#Mostrar histórico
history_df = pd.DataFrame(history.history)
history_df['loss'].plot()
plt.show()