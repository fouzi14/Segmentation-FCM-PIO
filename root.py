from numpy import where, clip, logical_and, maximum, minimum, power, sin, abs, pi, sqrt, sign, ones, ptp, min, sum, array, ceil, multiply, mean
from numpy.random import uniform, random, normal, choice
from math import gamma
from copy import deepcopy


class Root:
    """  """

    ID_MIN_PROB = 0  # min problem


    ID_POS = 0  # Position
    ID_FIT = 1  # Fitness

    EPSILON = 10E-50



    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True):


        self.verbose = verbose
        self.obj_func = obj_func
        self.__check_parameters__(lb, ub)

        self.étiration, self.population = None, None
        self.solution, self.loss_train = None, []

    def __check_parameters__(self, lb, ub):

        if isinstance(lb, list) and isinstance(ub, list):
            if len(lb) == len(ub):

                    self.problem_size = len(lb)
                    self.lb = array(lb)
                    self.ub = array(ub)
            else:
                print("La limite inférieure et la limite supérieure doivent avoir la même longueur")
                exit(0)



    def create_solution(self, minmax=0):

        position = uniform(self.lb, self.ub)
        fitness = self.get_fitness_position(position=position, minmax=minmax)
        return [position, fitness]
        #pour maximiser le prblm esle maximisation de prblm
    def get_fitness_position(self, position=None, minmax=0):

        return self.obj_func(position) if minmax == 0 else 1.0 / (self.obj_func(position) + self.EPSILON)



    def get_global_best_solution(self, pos=None, id_fit=None, id_best=None):

        sorted_pos = sorted(pos, key=lambda temp: temp[id_fit])

        return deepcopy(sorted_pos[id_best])

        #tester le position ida kant bin 0-255 trodelna true wtkhali la position else ta3tina random bin 0-255
    def amend_position_random_faster(self, position=None):
        return where(logical_and(self.lb <= position, position <= self.ub), position, uniform(self.lb, self.ub))

    def update_global_best_solution(self, pos=None, id_best=None, g_best=None):

        sorted_pos = sorted(pos, key = lambda temp: temp[self.ID_FIT])
        current_best = sorted_pos[id_best]
        return deepcopy(current_best) if current_best[self.ID_FIT] < g_best[self.ID_FIT] else deepcopy(g_best)

