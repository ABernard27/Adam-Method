# %%
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from functions.comparizon import get_loss
import matplotlib.pyplot as plt


# %%

# Instantiate the MNIST dataset
dataset = MNIST(root='./data',
                train=True,
                download=True,
                transform=ToTensor())

# Extraire les caractéristiques et les étiquettes
data = DataLoader(dataset, batch_size=128, shuffle=True)

# %%

# Calcul et affichage pour la régression multinomial
err_mult = get_loss(data, 'multinomial')

plt.figure(figsize=(10, 5))
plt.plot(err_mult[0], color='red')
plt.plot(err_mult[1], color='darkcyan')
plt.plot(err_mult[2], color='purple')
plt.ylim(0, 2)
plt.xlim(0, 200)
plt.grid()
plt.legend(['Adam', 'RMSprop', 'Adagrad'])
plt.xlabel("Timestep")
plt.ylabel("Valeur de la négative log-vraisemblance")
plt.title("Comparaison des méthodes d'optimisation pour une régression multinomiale")
plt.show()
# %%
# Calcul et affichage pour le réseau de neurone
err_nn = get_loss(data, 'neural_network')

plt.figure(figsize=(10, 5))
plt.plot(err_nn[0], color='red')
plt.plot(err_nn[1], color='darkcyan')
plt.plot(err_nn[2], color='purple')
plt.ylim(0, 2)
plt.xlim(0, 200)
plt.grid()
plt.legend(['Adam', 'RMSprop', 'Adagrad'])
plt.xlabel("Timestep")
plt.ylabel("Valeur de la négative log-vraisemblance")
plt.title("Comparaison des méthodes d'optimisation sur un réseau de neurone")
plt.show()