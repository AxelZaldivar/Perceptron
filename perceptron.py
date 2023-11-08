import numpy as np

# Función de activación
def step_function(x):
    return 1 if x >= 0 else 0

# Función del perceptrón
def perceptron(input, weights, bias):
    linear_combination = np.dot(input, weights) + bias
    output = step_function(linear_combination)
    return output

# Función de entrenamiento del perceptrón
def train_perceptron(input_data, labels, learning_rate=0.1, epochs=20):
    weights = np.random.rand(2)
    bias = 1
    
    for epoch in range(epochs):
        for input, label in zip(input_data, labels):
            prediction = perceptron(input, weights, bias)
            weights += learning_rate * (label - prediction) * input
            bias += learning_rate * (label - prediction)
    return weights, bias

# Entrenamiento para compuerta lógica OR
input_data_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels_or = np.array([0, 1, 1, 1])
weights_or, bias_or = train_perceptron(input_data_or, labels_or)

# Entrenamiento para compuerta lógica AND
input_data_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels_and = np.array([0, 0, 0, 1])
weights_and, bias_and = train_perceptron(input_data_and, labels_and)

# Prueba del perceptrón entrenado para OR
print("Pesos OR:", weights_or)
print("Sesgo OR:", bias_or)
print("Resultado OR:")
for input in input_data_or:
    result = perceptron(input, weights_or, bias_or)
    print(f"{input} -> {result}")

# Prueba del perceptrón entrenado para AND
print("\nPesos AND:", weights_and)
print("Sesgo AND:", bias_and)
print("Resultado AND:")
for input in input_data_and:
    result = perceptron(input, weights_and, bias_and)
    print(f"{input} -> {result}")