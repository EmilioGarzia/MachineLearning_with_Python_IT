# Machine Learning 2.1: Regressione Lineare per la previsione dei dati
# Emilio Garzia, 2023
# Per maggiori informazioni sui cenni teorici e la natura dei dati consultare il README.md file

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

peso = np.array([65, 75, 60, 93, 66, 84, 67, 89, 71, 63, 51, 58, 67, 51, 60, 55, 55, 58, 76, 55, 79])
altezza = np.array([180, 180, 173, 193, 164, 188, 175, 170, 180, 170, 161, 167, 165, 153, 170, 165, 155, 170, 170, 165, 188])

#calcolo i parametri ideali della mia retta lineare
slope, intercept, r, p, std_err = stats.linregress(peso, altezza)

#predizione altezza
nuovo_peso1 = 65
nuova_altezza1 = slope*nuovo_peso1+intercept

#predizione peso (formula inversa)
nuova_altezza2 = 180
nuovo_peso2 = (nuova_altezza2 - intercept)/slope

plt.scatter(peso, altezza, c="black")               #scatter del dataset
plt.plot(peso, slope*peso+intercept, c="green")     #plot della retta di regressione lineare
plt.scatter(nuovo_peso1, nuova_altezza1, c="red")   #predizione della nuova altezza
plt.scatter(nuovo_peso2, nuova_altezza2, c="blue")  #predizione del nuovo peso
plt.title("ESERCIZIO: Regressione lineare")
plt.xlabel("peso (kg)")
plt.ylabel("altezza (cm)")
plt.show()