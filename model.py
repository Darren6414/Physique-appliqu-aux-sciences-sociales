from mesa import Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from agent import House, Criminal
import math
from statistics import mean
import random
import numpy as np


def get_mean_att(model):     #renvoit la moyenne des attractivités
    att = [house.att_t for house in model.house_schedule.agents]
    return mean(att)


def get_max_att(model):      #renvoit le max des attractivités
    att = [house.att_t for house in model.house_schedule.agents]
    return max(att)


def get_min_att(model):      #renvoit le min des attractivités
    att = [house.att_t for house in model.house_schedule.agents]
    return min(att)


def get_num_criminals(model):     #renvoit le nombre de criminels
    return model.num_agents

def get_num_burgles(model):       #renvoit le nombre total de cambriolage
    att = [house.crime_events for house in model.house_schedule.agents]
    return sum(att)


def get_att_map(model):
    # renvoit une matrice dont chaque coefficient est l'attractivité de la maison i,j
    crime_counts = np.zeros((model.grid.width, model.grid.height))

    for cell in model.grid.coord_iter():
        content, x, y = cell
        crimes = 0
        for row in content:
            if isinstance(row, House):
                crimes = row.att_t
                crime_counts[x][y] = crimes
    return crime_counts

def get_max_att_pos(model):
    #renvoit la position du max d'attractivité
    max_pos=()
    max_att=0
    for row in model.house_schedule.agents:
        if row.att_t > max_att:
            max_pos = row.pos
            max_att = row.att_t
    return max_pos


class BurglaryModel(Model):
    """
    Une classe notre modèle
    Attributs: 
    -----------
    N : int
        le nombre de criminels
    width : int
        largeur de la grille
    length : int
        hauteur de la grille
    omega : float
        fixe une échelle de temps sur laquelle les victimisations répétées sont les plus 
        susceptibles de se produire
    delta : float
        variation de temps    
    theta: float
        la quantité fixe que l'on ajoute à l'attractivité dynamique Bs à chaque fois 
        qu'une maison est cambriolée
    eta : float
        un paramètre compris entre zéro et l'unité qui mesure la valeur 
        des eﬀets de voisinage.
    """

    def __init__(self, N, width, height, delta, omega, theta, eta, gamma, space):
        self.num_agents = N
        self.grid = MultiGrid(width, height, False)
        self.width = width
        self.height = height
        self.houses = self.width * self.height
        self.schedule = SimultaneousActivation(self)
        self.house_schedule = SimultaneousActivation(self)
        self.delta = delta
        self.omega = omega
        self.theta = theta
        self.eta = eta
        self.kill_agents = []
        self.gamma = gamma
        self.gen_agent = 1 - math.exp(-self.gamma*self.delta)
        self.total_agents = self.num_agents
        self.space = space

        a_0 = 0.2
        # on place une maison par case de la grille
        for i in range(self.width):
            for j in range(self.height):
                num = (i,j)
                a = House(num, self, a_0, i, j, self.delta, self.omega, self.theta, self.eta, self.space)
                self.grid.place_agent(a, (a.x_point, a.y_point))
                self.house_schedule.add(a)

        # on place les criminels
        for k in range(self.num_agents):
            unique_id = "criminal" + str(k)
            criminal = Criminal(unique_id, self, self.width, self.height)
            self.grid.place_agent(criminal, (criminal.x_point, criminal.y_point))
            self.schedule.add(criminal)

        # on crée un data set sur le modèle
        self.datacollector = DataCollector(
            model_reporters={"Mean_Attractiveness": get_mean_att,
                             "Max_Attractiveness": get_max_att,
                             "Min_Attractiveness": get_min_att,
                             "CrimeEvents": get_num_burgles,
                             "Criminals": get_num_criminals,
                             "MaxPos": get_max_att_pos},
            agent_reporters={"Att": lambda x: x.att_t if x.unique_id[:1]!="c" else None})

    #ajoute un criminel
    def add_criminals(self):
        start_count = self.total_agents + 1
        for i in range(self.houses):
            y = random.random()
            if y < self.gen_agent:
                unique_id = "criminal" + str(start_count)
                criminal = Criminal(unique_id, self, self.width, self.height)
                self.grid.place_agent(criminal, (criminal.x_point, criminal.y_point))
                self.schedule.add(criminal)
                start_count = start_count + 1
                self.total_agents = start_count
                self.num_agents = self.num_agents + 1


    def step(self):
        self.datacollector.collect(self)
        # calcule les attractivités à l'étape suivante en fonction de l'étape précédente
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

        

        self.add_criminals()



