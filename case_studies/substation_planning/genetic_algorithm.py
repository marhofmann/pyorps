
import pandapower as pp
from pandapower.topology import create_nxgraph
import networkx as nx
import numpy as np
from numba import njit
from check_tree_topologies import net_valid
from topology_reconfiguration_random_trees import create_trees_with_cycle_switches


class GeneticAlgorithm(object):
    def __init__(self, net, population_size, tournament_size, nr_of_generations=100, crossover_rate=0.5,
                 mutation_rate=0.5, keep_best_of_all_time=True, keep_best_of_generation=True,
                 optimization_function=None, **net_valid_kwargs):
        self.net = net
        self.cycle_basis = None
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.nr_of_generations = nr_of_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.keep_best_of_all_time = keep_best_of_all_time
        self.keep_best_of_generation = keep_best_of_generation
        self.optimization_function = optimization_function
        self.occurrences = dict()
        self.best_of_generations = dict()
        self.all_generations = dict()
        self.all_time_best_individual = None
        self.all_time_best_fitness = None
        self.individuals_tested = 0
        self.net_valid_kwargs = net_valid_kwargs

    def set_all_time_best(self, individual, fitness):
        if self.all_time_best_fitness is None or self.all_time_best_fitness > fitness:
            print(f"\nNew best performer after {self.individuals_tested} individuals tested:"
                  f"\n{individual}\nWith fitness:\n {fitness}")
            self.all_time_best_fitness = fitness
            self.all_time_best_individual = individual

    def get_init_population(self, size, resistance_first=True):
        cycles, population = create_trees_with_cycle_switches(self.net, size, resistance_first)
        self.cycle_basis = [np.array(cycle) for cycle in cycles]
        return population

    def topology_valid(self):
        nxg = create_nxgraph(self.net, multi=False, calc_branch_impedances=False)

        try:
            nx.find_cycle(nxg)
            return False
        except nx.NetworkXNoCycle:
            pass
        connected = {n for bus in self.net.ext_grid.bus.values for n in nx.node_connected_component(nxg, bus)}
        if any(bus not in connected for bus in net.load.bus):
            return False
        else:
            return True

    def fitness(self, individual):
        if (get := tuple(individual)) in self.all_generations:
            if get in self.occurrences:
                self.occurrences[get] += 1
            else:
                self.occurrences[get] = 2
            return self.all_generations[get]

        self.individuals_tested += 1

        if individual.size > np.unique(individual).size:
            return np.nan

        self.net.switch.closed = True
        self.net.switch.loc[individual, "closed"] = False
        if not self.topology_valid():
            return np.nan

        pp.runpp(self.net)

        if not net_valid(self.net, **self.net_valid_kwargs):
            return np.nan

        return self.optimization_function(self.net)

    def selection(self):
        selection = np.zeros((self.population_size, len(self.cycle_basis)), dtype=int)
        saved_individuals = len(self.all_generations)
        if saved_individuals < self.population_size:
            if saved_individuals > 0:
                selection[:saved_individuals] = np.array(list(self.all_generations.keys()))
            selection[saved_individuals:] = self.get_init_population(self.population_size - saved_individuals, False)
            return selection
        else:
            all_generations = np.array(list(self.all_generations.keys()))
            fitness = np.array(list(self.all_generations.values()))
            indices = np.arange(0, all_generations.shape[0])
        for pop_idx in range(self.population_size):
            tournament = np.random.choice(indices, size=self.tournament_size)
            winner = np.argmin(fitness[tournament])
            selection[pop_idx] = all_generations[tournament][winner]
        return selection

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.randint(0, len(self.cycle_basis))
            child1 = np.concatenate([parent1[:alpha], parent2[alpha:]])
            child2 = np.concatenate([parent2[:alpha], parent1[alpha:]])
            return child1, child2
        else:
            return parent1, parent2

    def mutation_advanced(self, individual):
        unique, counts = np.unique(individual, return_counts=True)
        if individual.size > unique.size:
            counts = dict(zip(unique, counts))
        else:
            counts = None
        for beta, value in enumerate(individual):
            cycle = self.cycle_basis[beta]
            if counts:
                mutation_rate = self.mutation_rate * counts[value]
                calc_values = np.array([0 if c in individual else 1 for c in cycle])
                calc_sum = calc_values.sum()
                p = np.array([0 if c in individual else 1 / calc_sum for c in cycle])
                if p.sum() == 0:
                    p = np.repeat(1 / cycle.size, cycle.size)
            else:
                mutation_rate = self.mutation_rate
                p = np.repeat(1 / cycle.size, cycle.size)
            if np.random.random() < mutation_rate:
                mutation = np.random.choice(cycle, p=p)
                individual[beta] = mutation
        return individual

    def advanced_dynamic_mutation(self, individual, dynamic_mutation_rates):
        unique, counts = np.unique(individual, return_counts=True)
        if individual.size > unique.size:
            counts = dict(zip(unique, counts))
        else:
            counts = None
        for beta, value in enumerate(individual):
            cycle = self.cycle_basis[beta]
            dyn_mut_rates = dynamic_mutation_rates[beta]
            if counts:
                mutation_rate = dyn_mut_rates[value] * counts[value]
                calc_values = np.array([0 if c in individual else 1 for c in cycle])
                calc_sum = calc_values.sum()
                p = np.array([0 if c in individual else 1 / calc_sum for c in cycle])
                if p.sum() == 0:
                    p = np.repeat(1 / cycle.size, cycle.size)
            else:
                mutation_rate = dyn_mut_rates[value]
                p = np.repeat(1 / cycle.size, cycle.size)
            if np.random.random() < mutation_rate:
                mutation = np.random.choice(cycle, p=p)
                individual[beta] = mutation
        return individual

    def dynamic_mutation(self, individual, dynamic_mutation_rates):
        for beta, value in enumerate(individual):
            cycle = self.cycle_basis[beta]
            if np.random.random() < dynamic_mutation_rates[value]:
                mutation = np.random.choice(cycle)
                individual[beta] = mutation
        return individual

    def mutation(self, individual):
        for beta, value in enumerate(individual):
            cycle = self.cycle_basis[beta]
            if np.random.random() < self.mutation_rate:
                mutation = np.random.choice(cycle)
                individual[beta] = mutation
        return individual

    def save_generation(self, population, fitness_values):
        self.all_generations.update({tuple(indi): fit for indi, fit in zip(population, fitness_values)})

    def get_best_of_population(self, population):
        fitness_values = np.array([self.fitness(individual) for individual in population])
        mask = ~np.isnan(fitness_values)
        fitness_values = fitness_values[mask]
        if len(fitness_values) == 0:
            return None, None
        population = population[mask]
        best_fitness_idx = np.argmin(fitness_values)
        best_individual = population[best_fitness_idx]
        best_fitness_values = np.nanmin(fitness_values)
        self.save_generation(population, fitness_values)
        return best_fitness_values, best_individual

    def get_next_generation(self, population):
        #values, counts = np.unique(population, return_counts=True)
        #dynamic_mutation_rates = dict(zip(values, counts/counts.max()))
        dynamic_mutation_rates = []
        for pop in population.T:
            values, counts = np.unique(pop, return_counts=True)
            dynamic_mutation_rates.append(dict(zip(values, self.mutation_rate * counts/counts.max())))

        next_generation = np.zeros((self.population_size, len(self.cycle_basis)), dtype=int)
        for pop_idx in range(0, self.population_size, 2):
            parent1 = population[pop_idx]
            parent2 = population[pop_idx + 1]
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.advanced_dynamic_mutation(child1, dynamic_mutation_rates)
            child2 = self.advanced_dynamic_mutation(child2, dynamic_mutation_rates)
            #child1 = self.mutation_advanced(child1)
            #child2 = self.mutation_advanced(child2)
            next_generation[pop_idx] = child1
            next_generation[pop_idx + 1] = child2
        return next_generation

    def run(self):
        population = self.get_init_population(self.population_size)

        for generation in range(self.nr_of_generations):
            best_fitness_value, best_individual = self.get_best_of_population(population)
            if best_fitness_value is not None:
                t_individual = tuple(best_individual)
                self.best_of_generations[generation] = (t_individual, best_fitness_value)

                self.set_all_time_best(t_individual, best_fitness_value)
            else:
                t_individual = None

            population = self.selection()

            next_generation = self.get_next_generation(population)

            if self.keep_best_of_generation and best_individual is not None:
                next_generation[-1] = best_individual
            if (self.keep_best_of_all_time and
                    t_individual is not None and
                    t_individual != self.all_time_best_individual):
                next_generation[0] = self.all_time_best_individual

            population = next_generation

        print(f"Finished after {self.nr_of_generations} generations with {self.individuals_tested} individuals tested:")
        print(f"Best individual: {self.all_time_best_individual} with {self.all_time_best_fitness} MW")
        self.net.switch.closed = True
        self.net.switch.loc[self.all_time_best_individual, 'closed'] = False
        pp.runpp(net)

        most_common_count = (None, 0)
        for key, value in self.occurrences.items():
            if value > most_common_count[1]:
                most_common_count = (key, value)
        print(f"Most common count: {most_common_count[1]} for {most_common_count[0]}")


if __name__ == '__main__':
    from time import perf_counter
    #net = pp.from_json(r"C:\Users\mhnn82\Documents\3_python_projects\ITEE\third_semester\exercise_one\1-LV-semiurb4
    # --1-no_sw_modified.json")
    #net = pp.from_json(r"C:\Users\mhnn82\Documents\3_python_projects\ITEE\third_semester\exercise_two
    # \example_lv_two_substations.json")
    #define_cable_box_buses(net)
    #start_bf = perf_counter()
    #result_brute_force = optimize_topology_brute_force(net)
    #print(result_brute_force)
    #print(f"Time for brute force search: {perf_counter() - start_bf} s")
    """
    Switch option with lowest maximal line loading: (16, 4, 7, 12, 20)
    min_vm_pu             0.922531
    max_vm_pu             0.965000
    max_line_loading     51.897051
    max_trafo_loading    69.469146
    Name: (16, 4, 7, 12, 20), dtype: float64
    Time for brute force search: 207.65075570000045 s
    """
    np.random.seed(1234)
    start_ga = perf_counter()

    def max_line_loading(nw):
        return nw.res_line.loading_percent.max()

    def sum_losses(nw):
        return nw.res_line.pl_mw.sum()


    def s_ext_grid(net):
        return np.sqrt(net.res_ext_grid.p_mw.sum() ** 2 + net.res_ext_grid.q_mvar.sum() ** 2)


    net = pp.from_json(r"C:\Users\mhnn82\Documents\3_python_projects\pyorps\case_studies\substation_planning\substation_planning_pp_grid.json")

    ga = GeneticAlgorithm(net, population_size=10, tournament_size=25, nr_of_generations=500,
                          crossover_rate=0.1, mutation_rate=0.1, max_trafo_loading=110,
                          optimization_function=s_ext_grid)
    ga.run()
    print(f"Time for genetic algorithm {perf_counter() - start_ga} s")

