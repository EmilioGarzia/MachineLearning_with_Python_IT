# Machine Learning 1.1: Osservare e Distribuire Uniformemente i dati
# Emilio Garzia, 2023
# Per maggiori informazioni sui cenni teorici e la natura dei dati consultare il README.md file

import numpy as np
import matplotlib.pyplot as plt

dataset = np.random.normal(5.0, 1.0, 10000)

plt.hist(dataset, 100)
plt.show()