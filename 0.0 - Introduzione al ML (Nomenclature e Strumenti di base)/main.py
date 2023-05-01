# Machine Learning 0.0: Introduzione agli strumenti di base
# Emilio Garzia, 2023
# Per maggiori informazioni sui cenni teorici e la natura dei dati consultare il README.md file

import numpy as np
from scipy import stats

#simple dataset
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]

#mathematical tools
mean = np.mean(speed)
median = np.median(speed)
mode = stats.mode(speed)

#OUTPUT
print(mean)
print(median)
print(mode)