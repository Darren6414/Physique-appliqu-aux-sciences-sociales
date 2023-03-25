"""
Ici on va définir tous les agents de notre modèle, c'est à dire les maisons ainsi que les 
cambrioleurs.
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
    Attributes: 
    -----------
    unique_id : int
        son numéro
    attractiveness : float
        son attractivité pour les cambrioleurs 
    x_point, y_point : float, float
        sa localisation
    delta : float
        variation de temps
    omega : float
        fixe une échelle de temps sur laquelle les victimisations répétées sont les plus 
        susceptibles de se produire
    theta: float
        la quantité fixe que l'on ajoute à l'attractivité dynamique beta à chaque fois 
        qu'une maison est cambriolée
    mu : float
        un paramètre compris entre zéro et l'unité qui mesure la valeur 
        des eﬀets de voisinage.

    """
    def __init__(self, unique_id, model, attractiveness, x_point, y_point, delta, omega, theta, mu):
        super().__init__(unique_id, model)       
        self.x_point = x_point
        self.y_point = y_point
        self.omega = omega
        self.delta = delta
        self.theta = theta
        self.mu = mu
        self.attractiveness = attractiveness
        self.crime_events = 0
        self.crime_list = [0]
        self.beta = 0
        self.att_t = self.attractiveness + self.beta
        self.pst = 1 - np.exp(-self.att_t*self.delta)

    def burgle(self):
        """ 
        On ajoute 1 à crime events quand un crime est commit
        """
        self.crime_events = self.crime_events + 1

    def new_beta(self):
        """
        On cherche dans le modèle toutes les maisons voisines à la maison choisie 
        et ensuite on applique la formule (2-6) du support pour déterminer la nouvelle valeur 
        de beta à l'instant t+delta
        """
        voisins = self.model.grid.get_neighbors(pos=(self.x_point, self.y_point),
        moore=False, include_center=False, radius=1)
        b = 0
        for i in voisins:
            if isinstance(i, House): # on regarde si il s'agit d'une maison
                b = b + i.beta
        self._beta = (self.beta + (self.mu / 4) * (b - 4 * self.beta)) * (1 - self.omega * (
            self.delta)) + self.theta * self.crime_events

    def new_att(self):
        """
        On calcule la nouvelle valeur de att_t à l'instant t+delta
        """
        self._att_t = self.beta + self.attractiveness

    def new_pst(self):
        """
        On calcule la nouvelle valeur de pst à l'instant t+delta
        """
        self._pst = 1 - np.exp(-self.att_t * self.delta)


    def step(self):
        self.new_beta()
        self.new_att()
        self.new_pst()

    def new_values(self):
        """
        On remplace les anciennes valeurs à l'instant t par les nouvelles valeurs obtenues 
        à l'instant t+delta
        """
        self.att_t = self._att_t
        self.beta = self._beta
        self.pst = self._pst
        self.crime_events = 0


"""
Ici on construit la classe criminel, chaque criminel est généré de manière uniforme dans 
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
    attractiveness : float
        son attractivité pour les cambrioleurs 
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
        Réinitialise la valeur de self.decision à  0 si le cambrioleur décide de quitter la maison et 1 si il 
        décide de la cambrioler.
        """
        cell_contents = self.model.grid.get_cell_list_contents([(self.x_point, self.y_point)])
        # renvoie une liste des agents contenus dans le noeuds identifiés les noeuds dont le contenu est vide sont exclus.
        for i in cell_contents:
            if isinstance(i, House):   # on regarde si il s'agit d'un criminel
                p = i.pst
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

        # on ordonne move par probabilite croissant
        move = sorted(move, key=lambda i: i['prob'])
        
        # on cumule les probas afin de ne pas toujours choisir les maisons qui ont les plus grande attractivités
        cp = 0
        for i in range(len(move)):
            cp = cp + move[i]['prob']
            move[i]['prob'] = cp

        # on décide si on bouge ou pas
        p = random.random()
        for m in move:
            if p < m['prob']:
                move_dest = m['house']
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
                    self.model.kill_agents.append(self) # on enlève le cambrioleur du modèle
        else:
            self.move()

#House(5, 128, 128, 2, 5, 5, 5.6, 0.2, 5)