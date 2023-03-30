import model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
from agent import House

model = model.BurglaryModel(10, 100, 100, delta = 0.01, omega = .06, theta = 10, eta = 0.1, gamma = 0.04, space = 0.1)


if __name__ == '__main__':    
    for i in range(1000):   #on effectue 1000 itérations
        model.step()
        print(i)

    crime_counts = np.zeros((model.grid.width, model.grid.height))     #à l'issu des 1000 itérations on crée une matrice qui stocke le nombre de cambriolage pour chaque maison 
    for cell in model.grid.coord_iter():
        content, x, y = cell
        crimes = 0
        for row in content:
            if isinstance(row, House):
                crimes = row.att_t
                crime_counts[x][y] = crimes

    norm = colors.Normalize(vmin=0.2, vmax=(model.theta * 0.5))    #on plot une heatmap correspondant à cette matrice

    plt.imshow(crime_counts, interpolation='nearest', cmap=plt.get_cmap('seismic'), norm=norm)
    plt.colorbar()
    plt.show()

    
