'''
Ici on va définir tous les agents de notre modèle, c'est à dire les maisons ainsi que les 
cambrioleurs.

On crée une classe House pour représenter les maisons de notre modèle avec Mesa, chaque maison
a un numéro, son attraction pour les cambrioleurs, sa localisation,omega sets a time scale over 
which repeat victimizations are most likely to occur, theta la quantité fixe que l'on 
ajoute à l'attractivité dynamique Bs à chaque fois qu'une maison est cambrioleé

Avec p s(t)=1-exp(-As(t)*δt) ce qui est logique puisque quand A(s) est grand la probabilite de cambrioler
la maison est proche de 1
'''
from mesa import Agent
import math
import random
import numpy as np

class House(Agent):

    def __init__(self, unique_id, model, attractiveness, x_point, y_point, delta, omega, theta, mu):
        super().__init__(unique_id, model)
        self.attractiveness = attractiveness
        self.x_point = x_point
        self.y_point = y_point
        self.delta = delta
        self.omega = omega
        self.theta = theta
        self.mu = mu
        self.crime_events = 0
        self.crime_list = [0]
        self.beta = 0
        self.att_t = self.attractiveness + self.beta
        self.p_s = 1 - np.exp(-self.att_t*self.delta)

    def burgle(self):
        ''' 
        #On rajoute 1 à crime events quand un crime est commit
        '''
        self.crime_events = self.crime_events + 1

    def new_beta(self):
        '''
        #On cherche dans le modèle toutes les maisons voisines à la maison choisie 
        et ensuite on applique la formule 2-6 du support pour déterminer la nouvelle valeur de beta
        à l'instant t+delta
        '''
        voisins = self.model.grid.get_neighbors(pos=(self.x_point, self.y_point),
        moore=False, include_center=False, radius=1)
        b_n = 0
        for i in voisins:
            if isinstance(i, House): #on regarde si i s'agit d'une maison
                b_n = b_n + i.beta

        self._beta = (self.beta + (self.mu / 4) * (b_n - 4 * self.beta)) * (1 - self.omega * (
            self.delta)) + self.theta * self.crime_events

    def new_att(self):
        '''
        On calcule la nouvelle valeur de att_t à l'instant t+delta
        '''
        self._att_t = self.beta + self.attractiveness

    def new_p_s(self):
        '''
        On calcule la nouvelle valeur de p_s à l'instant t+delta
        '''
        self._p_s = 1 - np.exp(-self.att_t * self.delta)


    def step(self):
        self.new_beta()
        self.new_att()
        self.new_p_s()

    def new_values(self):
        '''
        On remplace les anciennes valeurs à l'instant t par les nouvelles valeurs obtenues 
        à l'instant t+delta
        '''
        self.att_t = self._att_t
        self.beta = self._beta
        self.p_s = self._p_s
        self.crime_events = 0


'''
Ici on construit la classe criminelle, chaque criminelle est généré de manière uniforme dans 
l'environnement et est représenté par ses coordonnées (x_point, y_point), on lui ajoute aussi 
son choix modeliser par decision valant 0 si il decide de ne pas commettre le crime et 1
quand il décide de le commettre
'''
class Criminal(Agent):
    def __init__(self, unique_id, model, width, height):
        super().__init__(unique_id, model)
        self.x_point = random.randint(0, width-1)
        self.y_point = random.randint(0, height-1)
        self.decision = 0  

    def burgle_decision(self):
        """Returns a list of the agents contained in the nodes identified
        in `cell_list`; nodes with empty content are excluded.
        """
        cell_contents = self.model.grid.get_cell_list_contents([(self.x_point, self.y_point)])

        for i in cell_contents:
            if isinstance(i, House):   # on regarde si il s'agit d'un criminel
                p = i.p_s
                y = random.random()    # on choisit arbitrairement un nombre entre 0 et 1
                if y < p:
                    self.decision = 1  # le crime est commit
                else:
                    self.decision = 0  # sinon on quitte les lieux

    def move(self):
        voisins = self.model.grid.get_neighbors(pos=(self.x_point, self.y_point), moore=False,
                                                  include_center=False, radius=1)
        # on calcule la somme des attractivités des maisons au voisinage de la maison où se trouve 
        # le criminel
        a_t_sum = 0
        for i in voisins:
            if isinstance(i, House):
                a_t_sum = a_t_sum + i.att_t

        # on calcule la probabilité d'aller dans chacune des maisons avec la formule donnée par 2-3
        move = []
        for i in voisins:
            if isinstance(i, House): # on vérifie qu'il s'agit bien d'une maison
                move_dict = {}
                move_dict['house'] = (i.x_point, i.y_point)
                move_dict['prob'] = i.att_t / a_t_sum
                move.append(move_dict)

        # bubble sort the houses on va ranger notre   move par probabilite croissant
        for i in range(len(move) - 1, 0, -1):
            for j in range(i):
                if move[j]['prob'] > move[j + 1]['prob']:
                    temp = move[j]
                    move[j] = move[j + 1]
                    move[j + 1] = temp

        # convert move probabilities to cumulative
        cum_prob = 0
        for i in range(len(move)):
            cum_prob = cum_prob + move[i]['prob']
            move[i]['prob'] = cum_prob

        # now decide what house you'll move to
        p = random.random()

        for row in move:
            if p < row['prob']:
                move_dest = row['house']
                self.model.grid.move_agent(self, pos=move_dest)
                break
            else:
                continue

    def step(self):
        self.burgle_decision()

    def advance(self):
        if self.decision == 1:
            cell_contents = self.model.grid.get_cell_list_contents([(self.x_point, self.y_point)])
            for i in cell_contents:
                if isinstance(i, House):
                    i.burgle()
                    self.model.kill_agents.append(self)
        else:
            self.move()

print(House(5, 128, 128, 2, 5, 5, 5.6, 0.2, 5))