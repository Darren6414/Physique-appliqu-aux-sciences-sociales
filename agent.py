"""
Ici on va définir tous les agents de notre modèle, c'est à dire les maisons ainsi que les cambrioleurs.
"""
"""
Avec p s(t)=1-exp(-As(t)*δt) ce qui est logique puisque quand A(s) est grand la probabilite de cambrioler
la maison est proche de 1
"""
from mesa import Agent
import math
import random
import numpy as np


class House(Agent):
    """
    Une classe représentant chaque maison de notre modèle avec Mesa.
    Attributs: 
    -----------
    unique_id : int
        son numéro
    attractiveness: float
        la composante statique de son attractivité pour les cambrioleurs 
    x_point, y_point: float, float
        sa localisation
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

    def __init__(self, unique_id, model, attractiveness, x_point, y_point, delta, omega, theta, mu, space):
        super().__init__(unique_id, model)
        self.attractiveness = attractiveness
        self.x_point = x_point
        self.y_point = y_point
        self.delta = delta
        self.omega = omega
        self.theta = theta
        self.mu = mu
        self.space = space
        self.crime_events = 0      # Es(t) de notre modèle donnée par l'équation 2-4
        self.crime_list = [0]
        self.beta = 0
        self.att_t = self.attractiveness + self.beta
        self.p_s = 1 - math.exp(-self.att_t*self.delta)    # Es(t) de notre modèle donnée par l'équation 2-4

    def burgle(self):
        """ 
        On ajoute 1 à crime_events quand un crime est commit
        """
        self.crime_events = self.crime_events + 1

    def update_beta(self):
        """
        On cherche dans le modèle toutes les maisons voisines à la maison choisie 
        et ensuite on applique la formule (2-6) du support pour déterminer la nouvelle valeur 
        de Bs à l'instant t+delta
        """
        neighbors = self.model.grid.get_neighbors(pos=(self.x_point, self.y_point),
                                                  moore=False, include_center=False, radius=1)
        b_n = 0
        for i in neighbors:
            if isinstance(i, House):     # on regarde si il s'agit d'une maison
                b_n = b_n + i.beta

        self._beta = (self.beta + (self.mu / 4) * (b_n -4 * self.beta)) * (1 - self.omega * (        # on calcule laplacien*(l**2) avec la formule 2-7
            self.delta)) + self.theta * self.crime_events

    def update_att(self):
        """
        On calcule la valeur de As 
        """
        self._att_t = self.beta + self.attractiveness

    def update_p_s(self):
        """
        On calcule la valeur de proba 
        """
        self._p_s = 1 - np.exp(-self.att_t * self.delta)


    def step(self):
        """
        Etat du système à t
        """
        self.update_beta()
        self.update_att()
        self.update_p_s()

    def advance(self):
        """
        On remplace les anciennes valeurs à l'instant t par les nouvelles valeurs obtenues 
        à l'instant t+delta et on réinitialise le nombre de crimes
        """
        self.att_t = self._att_t
        self.beta = self._beta
        self.p_s = self._p_s
        self.crime_events = 0

"""
Ici on construit la classe criminal, chaque criminel est généré de manière uniforme dans 
l'environnement et est représenté par ses coordonnées (x_point, y_point), on lui ajoute aussi 
son choix modeliser par decision valant 0 si il decide de ne pas commettre le crime et 1
quand il décide de le commettre
"""

class Criminal(Agent):
    """
    Une classe représentant chaque cambrioleur.
    Attributes: 
    -----------
    unique_id : int
        son numéro
    width, height : float, float
        dimensions de l'espace
    """
    def __init__(self, unique_id, model, width, height):
        super().__init__(unique_id, model)
        self.x_point = random.randint(0, width-1)
        self.y_point = random.randint(0, height-1)
        self.decision = 0  

    def burgle_decision(self):
        """
        Attribut 0 comme valeur de self.decision si le cambrioleur décide de quitter la maison et 1 si il 
        décide de la cambrioler.
        """
        loc_contents = self.model.grid.get_cell_list_contents([(self.x_point, self.y_point)])
        # renvoie une liste des agents contenus dans la cellule identifiés les cellules  dont le contenu est vide sont exclus.
        for i in loc_contents:
            if isinstance(i, House):   # on regarde si il s'agit d'un criminel
                p = i.p_s    
                y = random.random()    # on choisit arbitrairement un nombre entre 0 et 1
                if y < p:
                    self.decision = 1  # le crime est commit
                else:
                    self.decision = 0  # sinon on quitte les lieux

    def move(self):
        """
        Renvoie 0 si le cambrioleur décide de quitter la maison et 1 si il 
        décide de la cambrioler.
        """
        neighbors = self.model.grid.get_neighbors(pos=(self.x_point, self.y_point), moore=False,
                                                  include_center=False, radius=1)
        # on calcule la somme des attractivités des maisons au voisinage de la maison où se trouve 
        # le criminel
        a_t_sum = 0
        for i in neighbors:
            if isinstance(i, House):
                a_t_sum = a_t_sum + i.att_t

        # on calcule la probabilité d'aller dans chacune des maisons avec la formule donnée par 2-3
        move_list = []
        for i in neighbors:
            if isinstance(i, House): # on vérifie qu'il s'agit bien d'une maison
                move_dict = {}
                move_dict['house'] = (i.x_point, i.y_point)
                move_dict['prob'] = i.att_t / a_t_sum
                move_list.append(move_dict)

        # on ordonne move par probabilite croissant
        for i in range(len(move_list) - 1, 0, -1):
            for j in range(i):
                if move_list[j]['prob'] > move_list[j + 1]['prob']:
                    temp = move_list[j]
                    move_list[j] = move_list[j + 1]
                    move_list[j + 1] = temp

        # on cumule les probas afin de ne pas toujours choisir les maisons qui ont les plus grande attractivités
        cum_prob = 0
        for i in range(len(move_list)):
            cum_prob = cum_prob + move_list[i]['prob']
            move_list[i]['prob'] = cum_prob

        # on décide si on bouge ou pas
        p = random.random()

        for row in move_list:
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
            loc_contents = self.model.grid.get_cell_list_contents([(self.x_point, self.y_point)])
            for i in loc_contents:
                if isinstance(i, House):
                    i.burgle()
                    self.model.kill_agents.append(self)      # on enlève le cambrioleur du modèle
        else:
            self.move()
