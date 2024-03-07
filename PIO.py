
from numpy.random import uniform
from numpy import exp, sum
from root import Root


class PIO(Root):


    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, étiration=None, population=100, R=0.2):
        Root.__init__(self, obj_func, lb, ub, verbose)
        self.étiration = étiration
        self.population = population
        self.R = R


    @property

    def train(self):
        pos = [self.create_solution() for _ in range(self.population)]
        g_best = self.get_global_best_solution(pos=pos, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        list_velocity = uniform(self.lb, self.ub, (self.population, self.problem_size))
        n_p = int(self.population / 2)

        for étiration in range(0, self.étiration):

            if étiration :  #  Opérateur ta3 carte et de boussole
                for i in range(0, self.population):
                    #v_new : la vitesse ta3 pigeon
                    v_new = list_velocity[i] * exp(-self.R * (étiration + 1)) + uniform() * (g_best[self.ID_POS] - pos[i][self.ID_POS])
                   #x_new : la position ta3 les pigeon
                    x_new = pos[i][self.ID_POS] + v_new
                    x_new = self.amend_position_random_faster(x_new)
                    fit = self.get_fitness_position(x_new)
                    if fit < pos[i][self.ID_FIT]:
                        pos[i] = [x_new, fit]
                        list_velocity[i] = v_new

            else:  # Opérateur de référence
                pos = sorted(pos, key=lambda item: item[self.ID_FIT])
                list_fit = [pos[i][self.ID_FIT] for i in range(0, n_p)]
                list_pos = [pos[i][self.ID_FIT] for i in range(0, n_p)]

                frac_up = sum([list_fit[i] * list_pos[i] for i in range(0, n_p)], axis=0)
                frac_down = n_p * sum(list_fit)
                # x_c : ta3 fitnesse
                x_c = frac_up / frac_down

                for i in range(0, self.population):
                    x_new = pos[i][self.ID_POS] + uniform() * (x_c - pos[i][self.ID_POS])
                    fit_new = self.get_fitness_position(x_new)
                    if fit_new < pos[i][self.ID_FIT]:
                        pos[i] = [x_new, fit_new]


            g_best = self.update_global_best_solution(pos, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Etiration: {}, Best fit: {}".format(étiration + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class NEXTPIO(PIO):


    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, étiration=750, population=100, R=0.2):
        PIO.__init__(self, obj_func, lb, ub, verbose, étiration, population, R)

    def train(self):
        pos = [self.create_solution() for _ in range(self.population)]
        g_best = self.get_global_best_solution(pos=pos, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)
        list_velocity = uniform(self.lb, self.ub, (self.population, self.problem_size))
        n_p = int(self.population / 2)

        for étiration in range(0, self.étiration):

            if étiration :
                for i in range(0, self.population):
                    v_new = list_velocity[i] * exp(-self.R * (étiration + 1)) + uniform() * (g_best[self.ID_POS] - pos[i][self.ID_POS])
                    x_new = pos[i][self.ID_POS] + v_new
                    x_new = self.amend_position_random_faster(x_new)
                    fit_new = self.get_fitness_position(x_new)
                    if fit_new < pos[i][self.ID_FIT]:
                        pos[i] = [x_new, fit_new]
                        list_velocity[i] = v_new

            else:  # Landmark operations
                pos = sorted(pos, key=lambda item: item[self.ID_FIT])
                list_fit = [pos[i][self.ID_FIT] for i in range(0, n_p)]
                list_pos = [pos[i][self.ID_FIT] for i in range(0, n_p)]
                frac_up = sum([list_fit[i] * list_pos[i] for i in range(0, n_p)], axis=0)
                frac_down = n_p * sum(list_fit)
                x_c = frac_up / frac_down #

                ## Move all pigeon based on target x_c
                for i in range(0, self.population):
                    if uniform() < 0.5:
                        x_new = pos[i][self.ID_POS] + uniform() * (x_c - pos[i][self.ID_POS])
                    else:
                        x_new = self.levy_flight(étiration, pos[i][self.ID_POS], g_best[self.ID_POS])
                    fit_new = self.get_fitness_position(x_new)
                    if fit_new < pos[i][self.ID_FIT]:
                        pos[i] = [x_new, fit_new]

            # Update the global best
            g_best = self.update_global_best_solution(pos, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Etiration: {}, Best fit: {}".format(étiration + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
