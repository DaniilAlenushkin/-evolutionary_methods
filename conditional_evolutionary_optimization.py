from scipy.optimize import minimize
import numpy as np

def function_to_minimize(x1, x2, x3, x4, x5):
    return np.exp(x1 * x2 * x3 * x4 * x5)

def get_h1(x1, x2, x3, x4, x5):
    return pow(x1, 2) + pow(x2, 2) + pow(x3, 2) + pow(x4, 2) + pow(x5, 2) - 10

def get_h2(x2, x3, x4, x5):
    return x2 * x3 - 5 * x4 * x5

def get_h3(x1, x2):
    return pow(x1, 3) + pow(x2, 3) + 1

def get_penalty(x1, x2, x3, x4, x5):
    return max([0, abs(get_h1(x1, x2, x3, x4, x5)) - epsilon]) * w.get('h1') + \
           max([0, abs(get_h2(x2, x3, x4, x5)) - epsilon]) * w.get('h2') + \
           max([0, abs(get_h3(x1, x2)) - epsilon]) * w.get('h3') + \
           max([0, abs(x1) - 2.3]) * w.get('x1') + \
           max([0, abs(x2) - 2.3]) * w.get('x2') + \
           max([0, abs(x3) - 3.2]) * w.get('x3') + \
           max([0, abs(x4) - 3.2]) * w.get('x4') + \
           max([0, abs(x5) - 3.2]) * w.get('x5')

def get_initial_generation():
    result = []
    for _ in range(POPULATION_SIZE):
        sub_result = []
        for _ in range(CHROMOSOMES_NUMBER):
            sub_result.append(''.join(list(map(lambda x: str(x),
                                               list(np.random.randint(2, size=GENS_NUMBER))))))
        result.append(sub_result)
    return result

def get_x_value_by_bin(individual):
    return [int(i, 2)*INTERVAL - (FUNCTION_DEFINITION_LENGTH/2) for i in individual]

def get_norm_vector(x_values):
    f_values = [function_to_minimize(i[0], i[1], i[2], i[3], i[4]) +
                get_penalty(i[0], i[1], i[2], i[3], i[4]) for i in x_values]
    pre_norm_vector = [1 / (d+1e-10) for d in f_values]
    norm_values = [i/sum(pre_norm_vector) for i in pre_norm_vector]
    return norm_values


def fitness(bin_population):
    x_values = list(map(get_x_value_by_bin, bin_population))
    norm_vector = get_norm_vector(x_values)
    return norm_vector

def reproduction(normalized_vectors, bin_population):
    new_population = []
    cum_weights = np.cumsum(normalized_vectors)
    spin = np.random.random(size=POPULATION_SIZE) * cum_weights[-1]
    indices = np.searchsorted(cum_weights, spin)
    new_population.extend([bin_population[i] for i in list(indices)])
    return new_population

def crossing_over(bin_population):
    np.random.shuffle(bin_population)
    for pair in range(0, len(bin_population), 2):
        if np.random.random() < CROSSING_OVER_PROBABILITY:
            number_of_chromosomes_to_exchange = np.random.randint(0, CHROMOSOMES_NUMBER)
            first_parent_chromosomes_index = [i for i in range(CHROMOSOMES_NUMBER)]
            np.random.shuffle(first_parent_chromosomes_index)

            second_parent_chromosomes_index = [i for i in range(CHROMOSOMES_NUMBER)]
            np.random.shuffle(second_parent_chromosomes_index)

            for chromosome_number in range(number_of_chromosomes_to_exchange):
                number_of_genes_to_exchange = np.random.randint(0, GENS_NUMBER)
                direction = np.random.randint(2)
                if direction:
                    face_1 = bin_population[pair][first_parent_chromosomes_index[chromosome_number]][:number_of_genes_to_exchange]
                    face_2 = bin_population[pair + 1][second_parent_chromosomes_index[chromosome_number]][:number_of_genes_to_exchange]
                    bin_population[pair][first_parent_chromosomes_index[chromosome_number]] = \
                        face_2 + bin_population[pair][first_parent_chromosomes_index[chromosome_number]][number_of_genes_to_exchange:]
                    bin_population[pair + 1][second_parent_chromosomes_index[chromosome_number]] = \
                        face_1 + bin_population[pair + 1][second_parent_chromosomes_index[chromosome_number]][number_of_genes_to_exchange:]
                else:
                    back_1 = bin_population[pair][first_parent_chromosomes_index[chromosome_number]][
                             len(bin_population[pair][first_parent_chromosomes_index[chromosome_number]]) - number_of_genes_to_exchange:]
                    back_2 = bin_population[pair + 1][second_parent_chromosomes_index[chromosome_number]][
                             len(bin_population[pair][second_parent_chromosomes_index[chromosome_number]]) - number_of_genes_to_exchange:]
                    bin_population[pair][first_parent_chromosomes_index[chromosome_number]] = \
                        bin_population[pair][first_parent_chromosomes_index[chromosome_number]][
                        :len(bin_population[pair][first_parent_chromosomes_index[chromosome_number]]) - number_of_genes_to_exchange] + back_2
                    bin_population[pair + 1][second_parent_chromosomes_index[chromosome_number]] = \
                        bin_population[pair + 1][second_parent_chromosomes_index[chromosome_number]][
                        :len(bin_population[pair][second_parent_chromosomes_index[chromosome_number]]) - number_of_genes_to_exchange] + back_1
    return bin_population

def mutation(bin_population):
    for individual in range(len(bin_population)):
        for chromosome in range(len(bin_population[individual])):
            if np.random.random() < MUTATION_PROBABILITY:
                number_of_gen = np.random.randint(len(bin_population[individual][chromosome]))
                bin_population[individual][chromosome] = bin_population[individual][chromosome][:number_of_gen] + \
                                    str((int(bin_population[individual][chromosome][number_of_gen]) + 1) % 2) + \
                                    bin_population[individual][chromosome][number_of_gen + 1:]
    return bin_population

"""
def print_information(bin_population, title):
    x_values = list(map(get_x_value_by_bin, bin_population))
    f_values = [function_to_minimize(i[0], i[1], i[2], i[3], i[4]) for i in x_values]
    best_x1 = x_values[f_values.index(min(f_values))][0]
    best_x2 = x_values[f_values.index(min(f_values))][1]
    best_x3 = x_values[f_values.index(min(f_values))][2]
    best_x4 = x_values[f_values.index(min(f_values))][3]
    best_x5 = x_values[f_values.index(min(f_values))][4]
    print(title)
    print(f'Minimum point its f({best_x1}, {best_x2}, {best_x3}, {best_x4}, {best_x5}) = {min(f_values)}')
    print(f'h1(x) = {get_h1(best_x1, best_x2, best_x3, best_x4, best_x5)}')
    print(f'h2(x) = {get_h2(best_x2, best_x3, best_x4, best_x5)}')
    print(f'h3(x) = {get_h3(best_x1, best_x2)}')
    print()
"""

def print_information(bin_population, title):
    x_values = list(map(get_x_value_by_bin, bin_population))
    f_values = [function_to_minimize(i[0], i[1], i[2], i[3], i[4]) for i in x_values]
    h1_values = [get_h1(i[0], i[1], i[2], i[3], i[4]) for i in x_values]
    h2_values = [get_h2(i[1], i[2], i[3], i[4]) for i in x_values]
    h3_values = [get_h3(i[0], i[1]) for i in x_values]
    result = [abs(f_values[i]) + abs(h1_values[i]) + abs(h2_values[i]) + abs(h3_values[i]) for i in range(len(x_values))]
    best_individual = result.index(min(result))
    best_x1 = x_values[best_individual][0]
    best_x2 = x_values[best_individual][1]
    best_x3 = x_values[best_individual][2]
    best_x4 = x_values[best_individual][3]
    best_x5 = x_values[best_individual][4]

    print(title)
    print(f'Minimum point its f({best_x1}, {best_x2}, {best_x3}, {best_x4}, {best_x5}) = '
          f'{function_to_minimize(best_x1, best_x2, best_x3, best_x4, best_x5)}')
    print(f'h1(x) = {get_h1(best_x1, best_x2, best_x3, best_x4, best_x5)}')
    print(f'h2(x) = {get_h2(best_x2, best_x3, best_x4, best_x5)}')
    print(f'h3(x) = {get_h3(best_x1, best_x2)}')
    print()


if __name__ == '__main__':
    ACCURACY = 2
    POPULATION_SIZE = 2500
    NUMBER_OF_GENERATIONS = 500
    MUTATION_PROBABILITY = 0.01
    CROSSING_OVER_PROBABILITY = 0.5
    FUNCTION_DEFINITION_LENGTH = 8  # [-4; 4]
    CHROMOSOMES_NUMBER = 5
    GENS_NUMBER = (int(np.ceil(np.log2(FUNCTION_DEFINITION_LENGTH * 10 ** ACCURACY))))
    INTERVAL = FUNCTION_DEFINITION_LENGTH / 2 ** GENS_NUMBER
    TITLES = ['First generation: ', 'Middle generation: ', 'Latest generation: ']
    w = {'h1': 100,
         'h2': 100,
         'h3': 100,
         'x1': 50,
         'x2': 50,
         'x3': 50,
         'x4': 50,
         'x5': 50}

    epsilon = 0.0001
    counter_titles = 0
    population = get_initial_generation()
    for generation in range(NUMBER_OF_GENERATIONS):
        norm_vectors = fitness(population)
        population = reproduction(norm_vectors, population)
        population = crossing_over(population)
        population = mutation(population)
        if generation in [0, int(NUMBER_OF_GENERATIONS / 2), NUMBER_OF_GENERATIONS - 1]:
            print_information(population, TITLES[counter_titles])
            counter_titles += 1
