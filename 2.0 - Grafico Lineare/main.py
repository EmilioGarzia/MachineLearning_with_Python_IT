# Machine Learning 2.0: Grafico Lineare
# Emilio Garzia, 2023
# Per maggiori informazioni sui cenni teorici e la natura dei dati consultare il README.md file

import matplotlib.pyplot as plt

#stabilisco i valori della pendenza e della intercetta
slope = 1.2
intercept = 7

x = list()
y = list()

#definisco i valori della funzione lineare
for i in range(11):
    x.append(i)
    y.append(i * slope + intercept)

#mando in output il grafico della mia funzione lineare
plt.plot(x,y)
plt.title("y=ax+b slope=1.2 intercept=7")
plt.xlabel("x")
plt.ylabel("y")
plt.show()