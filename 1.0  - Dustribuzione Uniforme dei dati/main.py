# Machine Learning 1.0: Osservare e Distribuire Uniformemente i dati
# Emilio Garzia, 2023
# Per maggiori informazioni sui cenni teorici e la natura dei dati consultare il README.md file

import numpy as np
import matplotlib.pyplot as plt

#genero il mio dataset
dataset = np.random.normal(0.0, 5.0, 1000)

#mando a schermo il plot con i dati in un istogramma con 100 barre
plt.hist(dataset, 100)
plt.show()