# Machine Learning 2.2: Regressione Polinomiale
# Emilio Garzia, 2023
# Per maggiori informazioni sui cenni teorici e la natura dei dati consultare il README.md file

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#dataset
time = np.array([1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22])
speed = np.array([100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100])

#regressione polinomiale ideale
mymodel = np.poly1d(np.polyfit(time, speed, 3))
myline = np.linspace(1, 22, 100)

#regressione polinomiale con overfitting
wrongmodel = np.poly1d(np.polyfit(time, speed, 2))

#predire la velocità di un veicolo che attraversa il casello alle 17:00
prediction = mymodel(17)

#posizionamento dei dati sul grafico
plt.scatter(time, speed, c="black")                        #scatter del dataset
plt.plot(myline, mymodel(myline), c="lime")     #regressione polinomiale ideale
plt.plot(myline, wrongmodel(myline), c="red")   #overfitting
plt.scatter(17, prediction, c="blue")          #previsione con regressione polinomiale

#settaggi estetici del grafico
myFont = {"family":"Cambria", "size": 16, "color":"red", "weight":"bold", "style":"italic"}
labels = ["Dati", "Target", "Linea di fitting [M=2]", "Previsione velocità alle 17:00"]
plt.xlabel("time")
plt.ylabel("speed (Km/h)")
plt.xticks(np.arange(1,23))
plt.title("M=2", fontdict=myFont)
plt.legend(labels)
plt.show()