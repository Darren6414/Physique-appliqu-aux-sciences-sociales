"""
Ici on va définir tous les agents de notre modèle, c'est à dire les maisons ainsi que les cambrioleurs.
"""
"""
Avec p s(t)=1-exp(-As(t)*δt) ce qui est logique puisque quand A(s) est grand la probabilite de cambrioler
la maison est proche de 1
"""
from mesa import Agent
import random
import numpy as np

class House(Agent):
    """
    Une classe représentant chaque maison de notre modèle avec Mesa.
    Attributs: 
    -----------
    unique_id : int
        son numéro
    attract: float
        la composante statique de son attractivité pour les cambrioleurs 
    x, y: float, float
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
    def __init__(self, unique_id, model, attract, x, y, delta, omega, theta, eta, space):
        super().__init__(unique_id, model)       
        self.attract = attract
        self.x = x
        self.y = y
        self.delta = delta
        self.omega = omega
        self.theta = theta
        self.eta = eta
        self.space = space
        self.crimes = 0                                 # Es(t) de notre modèle donnée par l'équation 2-4
        self.crime_liste = [0]
        self.Bs = 0
        self.As = self.attract + self.Bs
        self.proba = 1 - np.exp(-self.As*self.delta)    # voir equation 2-2

    def burgle(self):
        """ 
        On ajoute 1 à crime events quand un crime est commit
        """
        self.crimes = self.crimes + 1

    def new_Bs(self):
        """
        On cherche dans le modèle toutes les maisons voisines à la maison choisie 
        et ensuite on applique la formule (2-6) du support pour déterminer la nouvelle valeur 
        de Bs à l'instant t+delta
        """
        voisins = self.model.grid.get_neighbors(pos=(self.x, self.y),moore=False, include_center=False, radius=1)
        Bsv = 0
        if isinstance(v, House):                        # on regarde si il s'agit d'une maison
            Bsv = np.sum([v.Bs for v in voisins])
  
        laplacien = Bsv - (4 * self.Bs)                 # on calcule laplacien*(l**2) avec la formule 2-7
        self.nBs = (self.Bs + (self.eta / 4) * laplacien) * (1 - self.omega * (self.delta)) + self.theta * self.crimes

    def new_As(self):
        """
        On calcule la valeur de As 
        """
        self.nAs = self.Bs + self.attract

    def new_proba(self):
        """
        On calcule la valeur de proba 
        """
        self.nproba = 1 - np.exp(-self.As * self.delta)


    def step(self):
        """
        Etat du système à t
        """
        self.new_Bs()
        self.new_As()
        self.new_proba()

    def new_values(self):
        """
        On remplace les anciennes valeurs à l'instant t par les nouvelles valeurs obtenues 
        à l'instant t+delta et on réinitialise le nombre de crimes
        """
        self.As = self.nAs
        self.Bs = self.nBs
        self.proba = self.nproba
        self.crimes = 0


"""
Ici on construit la classe criminel, chaque criminel est généré de manière uniforme dans 
l'environnement et est représenté par ses coordonnées (x_point, y_point), on lui ajoute aussi 
son choix modeliser par decision valant 0 si il decide de ne pas commettre le crime et 1
quand il décide de le commettre
"""
class Criminel(Agent):
    """
    Une classe représentant chaque cambrioleur.
    Attributes: 
    -----------
    unique_id : int
        son numéro
    attract : float
        son attractivité pour les cambrioleurs 
    longueur, largeur : float, float
        dimensions de l'espace
    """
    def __init__(self, unique_id, model, largeur, longueur):
        super().__init__(unique_id, model)
        self.x = random.randint(0, largeur-1)
        self.y = random.randint(0, longueur-1)
        self.choix = 0  

    def burgle_choix(self):
        """
        Attribut 0 comme valeur de self.decision si le cambrioleur décide de quitter la maison et 1 si il 
        décide de la cambrioler.
        """
        cell_contents = self.model.grid.get_cell_list_contents([(self.x, self.y)])
        # renvoie une liste des agents contenus dans la cellule identifiés les cellules  dont le contenu est vide sont exclus.
        for i in cell_contents:
            if isinstance(i, House):   # on regarde si il s'agit d'un criminel
                p = i.proba
                p_prime = random.random()    # on choisit arbitrairement un nombre entre 0 et 1
                if p_prime < p:
                    self.choix = 1  # le crime est commit
                else:
                    self.choix = 0  # sinon on quitte les lieux

    def move(self):
        """
        Renvoie 0 si le cambrioleur décide de quitter la maison et 1 si il 
        décide de la cambrioler.
        """
        voisins = self.model.grid.get_neighbors(pos=(self.x, self.y), moore=False, include_center=False, radius=1)
        # on calcule la somme des attractivités des maisons au voisinage de la maison où se trouve 
        # le criminel
        Asum = 0
        A = [v.As  for v in voisins]
        if isinstance(v, House):
            Asum = sum(A)
            
        # on calcule la probabilité d'aller dans chacune des maisons avec la formule donnée par 2-3
        move = []
        for v in voisins:
            if isinstance(v, House): # on vérifie qu'il s'agit bien d'une maison
                move_dict = {}
                move_dict['house'] = (v.x, v.y) 
                move_dict['chance'] = v.As / Asum
                move.append(move_dict)

        # on ordonne move par probabilite croissant
        move = sorted(move, key=lambda i: i['chance'])
        
        # on cumule les probas afin de ne pas toujours choisir les maisons qui ont les plus grande attractivités
        cp = 0
        for i in range(len(move)):
            cp = cp + move[i]['chance']
            move[i]['chance'] = cp

        # on décide si on bouge ou pas
        p = random.random()
        for m in move:
            if p < m['chance']:
                move_dest = m['house']
                self.model.grid.move_agent(self, pos=move_dest)
                break
            else:
                continue

    def step(self):
        self.burgle_choix()

    def advance(self):
        if self.choix == 1:
            cell_contents = self.model.grid.get_cell_list_contents([(self.x, self.y)])
            for i in cell_contents:
                if isinstance(i, House):
                    i.burgle()
                    self.model.kill_agents.append(self) # on enlève le cambrioleur du modèle
        else:
            self.move()

#House(5, 128, 128, 2, 5, 5, 5.6, 0.2, 5)