# Machine Learning 4.0: Pattern Recognition
# Emilio Garzia, 2023
# Per maggiori informazioni sui cenni teorici e la natura dei dati consultare il README.md file

import matplotlib.pyplot as plt
import random

# Creo 500 valori casuali per i punti XY
numPoints = 500
xPoints = [random.uniform(0, 400) for i in range(numPoints)]
yPoints = [random.uniform(0, 400) for i in range(numPoints)]

# Definisco la funzione lineare
def f(x): return x * 1.2 + 50

# Stabiliso per ogni punto la classe di appartenenza
desired = [1 if yPoints[i] > f(xPoints[i]) else 0 for i in range(numPoints)]

# Aggiungo al grafico la retta lineare e i valori casuali
fig, ax = plt.subplots()
ax.set_ylim(0, 400)
ax.plot([0, 400], [f(0), f(400)], color='black')

# Coloro di nero i punti al di sopra della retta e di blu quelli al di sotto
for i in range(numPoints):
    color = 'blue' if desired[i] == 0 else 'black'
    ax.scatter(xPoints[i], yPoints[i], c=color)

plt.show()