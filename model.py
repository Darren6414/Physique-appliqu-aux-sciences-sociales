from mesa import Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from agent import House, Criminel
from statistics import mean, median
import random
import numpy as np


def As_moyenne(model):
    att = [house.As for house in model.house_schedule.agents]
    return median(att)


def As_maxi(model):
    att = [house.As for house in model.house_schedule.agents]
    return max(att)


def As_min(model):
    att = [house.As for house in model.house_schedule.agents]
    return min(att)


def id_criminels(model):
    return model.num_agents

def num_burgles(model):
    att = [house.crimes for house in model.house_schedule.agents]
    return sum(att)


def As_map(model):
    # create numpy matrix
    crime_counts = np.zeros((model.grid.largeur, model.grid.longueur))

    for cell in model.grid.coord_iter():
        content, x, y = cell
        crimes = 0
        for row in content:
            if isinstance(row, House):
                crimes = row.As
                crime_counts[x][y] = crimes
    return crime_counts

def As_maxi_pos(model):
    max_pos=()
    max_att=0
    for row in model.house_schedule.agents:
        if row.As > max_att:
            max_pos = row.pos
            max_att = row.As
    return max_pos


class BurglaryModel(Model):

    def __init__(self, N, largeur, longueur, b_rate, delta, omega, theta, mu, gamma, space):
        self.num_agents = N
        self.grid = MultiGrid(largeur, longueur, True)
        self.largeur = largeur
        self.longueur = longueur
        self.houses = self.largeur * self.longueur
        self.schedule = SimultaneousActivation(self)
        self.house_schedule = SimultaneousActivation(self)
        self.b_rate = b_rate
        self.delta = delta
        self.omega = omega
        self.theta = theta
        self.mu = mu
        self.kill_agents = []
        self.gamma = gamma
        self.gen_agent = 1 - np.exp(-self.gamma*self.delta)
        self.total_agents = self.num_agents
        self.space = space

        a_0 = 0.2
        # place houses on grid, 1 house per grid location
        for i in range(self.largeur):
            for j in range(self.longueur):
                num = str(i) + str(j)
                num = int(num)
                a = House(num, self, a_0, i, j, self.delta, self.omega, self.theta, self.mu, self.space)
                self.grid.place_agent(a, (a.x, a.y))
                self.house_schedule.add(a)

        # place the criminals
        for k in range(self.num_agents):
            unique_id = "criminel" + str(k)
            criminel = Criminel(unique_id, self, self.largeur,  self.longueur)
            self.grid.place_agent(criminel, (criminel.x, criminel.y))
            self.schedule.add(criminel)

        # set up data collection
        self.datacollector = DataCollector(
            model_reporters={"Moyenne_Attractivité": As_moyenne,
                             "Maximum_Attractivité": As_maxi,
                             "Minimum_Attractivité": As_min,
                             "Crimes": num_burgles,
                             "Criminels": id_criminels,
                             "MaxPos": As_maxi_pos},
            agent_reporters={"Att": lambda x: x.As if x.unique_id[:1]!="c" else None})


    def add_criminels(self):
        start = self.total_agents + 1
        for i in range(self.houses):
            y = random.random()
            if y < self.gen_agent:
                unique_id = "criminel" + str(start)
                criminel = Criminel(unique_id, self, self.largeur, self.longueur)
                self.grid.place_agent(criminel, (criminel.x, criminel.y))
                self.schedule.add(criminel)
                start = start + 1
                self.total_agents = start
                self.num_agents = self.num_agents + 1


    def step(self):
        self.datacollector.collect(self)
        # cycle through all houses and calculate updates on their attractiveness
        self.house_schedule.step()
        self.schedule.step()
        for row in self.kill_agents:
            try:
                self.grid.remove_agent(row)
                self.schedule.remove(row)
                self.kill_agents.remove(row)
                self.num_agents = self.num_agents - 1

            except:
                self.kill_agents.remove(row)

        # add new criminels

        self.add_criminels()


if __name__ == '__main__':
    model = BurglaryModel(5, 12, 1, 2, 5, 5, 5.6, 0.2, 5, 1)
    for i in range(10):
        model.step()

    #print(model)