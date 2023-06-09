import model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
from agent import House

model = model.BurglaryModel(0, 12, 10, 5, 0.01, .06, 5.6, 0.1, 0.019, 1)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    for i in range(1000):
        model.step()
        #print(i)

    crime_counts = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        content, x, y = cell
        crimes = 0
        for row in content:
            if isinstance(row, House):
                crimes = row.As
                crime_counts[x][y] = crimes

    norm = colors.Normalize(vmin=0.2, vmax=(model.theta * 0.75))

    plt.imshow(crime_counts, interpolation='nearest', cmap=plt.get_cmap('seismic'), norm=norm)
    plt.colorbar()
    plt.show()

    As = model.datacollector.get_model_vars_dataframe()
    houses = model.datacollector.get_agent_vars_dataframe()

    As.to_csv('model_stats.csv')
    houses.to_csv('houses.csv')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/