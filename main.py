import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

classification = pd.read_csv("wavelength.csv")

y = classification['Class']
X = classification[['R_10', 'R_8', 'R_6', 'G_10', 'G_8', 'G_6', 'B_10', 'B_8', 'B_6']]
X_new = classification[['R_8', 'G_8', 'B_8']]

#Define optimizers, activations, and test splits
optimizers = ['adam', 'sgd', 'lbfgs']
activations = ['relu', 'tanh', 'logistic', 'identity']
test_splits = [0.5, 0.4, 0.3, 0.2, 0.1]

best_accuracy = 0
best_combination = {}
results = {}

#Loop through all optimizers and activation functions to optimize model
for optimizer in optimizers:
    for activation_pick in activations:
        accuracies = []
        for split in test_splits:
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=split, random_state=int(split*10))
            model = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=300, activation=activation_pick, solver=optimizer, random_state=27)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))

        avg_accuracy = np.mean(accuracies)
        results[f"{optimizer}_{activation_pick}"] = avg_accuracy

        #Update best optimizer and activation if accuracy is best
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_combination = {
                'optimizer': optimizer,
                'activation': activation_pick,
                'accuracy': avg_accuracy
            }

#Print optimal combination and its R-squared score
print("\nOptimal Combination:")
print(f"Optimizer: {best_combination['optimizer'].upper()}")
print(f"Activation: {best_combination['activation']}")
print(f"Average Accuracy: {best_combination['accuracy']:.4f}")

#Retrieve user input for the three wavelengths to predict a material class
try:
    R_8 = float(input("Enter red channel wavelength for 8 mM concentration: "))
    G_8 = float(input("Enter green channel wavelength for 8 mM concentration: "))
    B_8 = float(input("Enter blue channel wavelength for 8 mM concentration: "))
except ValueError:
    print("Please enter a number value wavelength!")
    exit()

X_input = np.array([R_8, G_8, B_8]).reshape(1, -1)
prediction = model.predict(X_input)
print("The predicted material class is:", prediction[0])




