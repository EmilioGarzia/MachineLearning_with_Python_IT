# Machine Learning 0.1: Calcolo dei percentili
# Emilio Garzia, 2023
# Per maggiori informazioni sui cenni teorici e la natura dei dati consultare il README.md file

import numpy as np

ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

percent_50 = np.percentile(ages, 50)

print(percent_50)

# OUTPUT: 31.0
# Quindi, il 50% delle persone ha un età di 31 anni o inferiore