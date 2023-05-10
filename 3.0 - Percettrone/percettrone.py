# Machine Learning: Implementazione di un percettrone
# Emilio Garzia, 2023
# Per maggiori informazioni sui cenni teorici e la natura dei dati consultare il README.md file

#Funzione per il calcolo della somma ponderata
def somma_ponderata(inputs, pesi):
    prodotti = list()
    for x, y in zip(inputs, pesi):
        prodotti.append(x*y)
    return sum(prodotti)

#Dati del problema
soglia = 1.5
inputs = [1, 0, 1, 0, 1]
pesi = [0.7, 0.6, 0.5, 0.3, 0.4]

#calcolo della somma ponderata
output_percettrone = somma_ponderata(inputs, pesi)

#output del percettrone
previsione = True if output_percettrone > soglia else False

#stampa della soluzione
print("Devo andare al concerto? -> " + str(previsione))