from init_path import one_parent_init
one_parent_init()
import commons.my_dataset as my_dataset
import numpy as np
from keras import Sequential
from keras.api.layers import Dense
import commons.plot_utils as plot_utils

m = 100
X,Y = my_dataset.get_beans1(m)
plot_utils.show_scatter(X, Y)

print(X)
print(Y)


model = Sequential()
model.add(Dense(units=1, activation="sigmoid", input_dim=1))

model.compile(loss="mean_squared_error", optimizer="sgd", metrics=['accuracy'])

model.fit(X, Y, epochs=5000, batch_size=10)

pres = model.predict(X)

plot_utils.show_scatter_curve(X, Y, pres)

