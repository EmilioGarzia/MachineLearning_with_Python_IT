# Machine Learning 2.3: Regressione Multipla
# Emilio Garzia, 2023
# Per maggiori informazioni sui cenni teorici e la natura dei dati consultare il README.md file

import pandas
from sklearn import linear_model

#leggo le informazioni dal file CSV
dataset = pandas.read_csv("dataset.csv")

#definisco le variabili indipendenti (X) e la variabile dipendente (y)
X = dataset[["Weight", "Volume"]]
y = dataset[["CO2"]]

#applico la regressione multipla e faccio un fitting dei dati
regression = linear_model.LinearRegression()
regression.fit(X, y)
print(regression)
#predico un valore di CO2 in base a due variabili
predictionCO2 = regression.predict([[2300,1300]])
print(predictionCO2)