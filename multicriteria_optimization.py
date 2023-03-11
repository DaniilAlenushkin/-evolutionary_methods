import matplotlib.pyplot as plt
import numpy as np


def get_value_f1(x1, x2):
    return (((x1-2)**2)/2) + (((x2+1)**2)/13) + 3

def get_value_f2(x1, x2):
    return (((x1+x2-3)**2)/36) + (((-x1+x2+2)**2)/8) - 17

def get_value_f3(x1, x2):
    return ((((3*x1)- (2*x2)-1)**2)/175) + (((-x1+(2*x2))**2)/17) - 13

def get_x_value_by_bin(individual):
    return [int(i, 2)*INTERVAL - (FUNCTION_DEFINITION_LENGTH/2) for i in individual]

def get_initial_generation():
    result = []
    for _ in range(POPULATION_SIZE):
        sub_result = []
        for _ in range(GENS_NUMBER):
            sub_result.append(''.join(list(map(lambda x: str(x),
                                               list(np.random.randint(2, size=CHROMOSOMES_NUMBER))))))
        result.append(sub_result)
    return result

def get_norm_vector(x_values, function):
    f_values = [function(i[0], i[1]) for i in x_values]
    pre_norm_vector_by_f = [1  - (i/sum(f_values)) for i in f_values]
    norm_vector_by_f = [(i/sum(pre_norm_vector_by_f)) for i in pre_norm_vector_by_f]
    return norm_vector_by_f

def fitness(bin_population):
    x_values = list(map(get_x_value_by_bin, bin_population))
    norm_vector_by_f1 = get_norm_vector(x_values, get_value_f1)
    norm_vector_by_f2 = get_norm_vector(x_values, get_value_f2)
    norm_vector_by_f3 =get_norm_vector(x_values, get_value_f3)
    return [norm_vector_by_f1, norm_vector_by_f2, norm_vector_by_f3]

def reproduction(normalized_vectors, bin_population):
    new_population = []
    for vector in normalized_vectors:
        cum_weights = np.cumsum(vector)
        spin = np.random.random(size=int(POPULATION_SIZE/3)) * cum_weights[-1]
        indices = np.searchsorted(cum_weights, spin)
        new_population.extend([bin_population[i] for i in list(indices)])
    np.random.shuffle(new_population)
    return new_population

def crossing_over(bin_population):
    np.random.shuffle(bin_population)
    for pair in range(0, len(bin_population), 2):
        if  np.random.random() < CROSSING_OVER_PROBABILITY:
            if np.random.randint(2):
                first_parent_indexes_gen = [0, 1]
                np.random.shuffle(first_parent_indexes_gen)

                second_parent_indexes_gen = [0, 1]
                np.random.shuffle(second_parent_indexes_gen)
            else:
                first_parent_indexes_gen = [np.random.randint(2)]
                second_parent_indexes_gen = [np.random.randint(2)]

            for gen_index in range(len(first_parent_indexes_gen)):
                number_of_genes = np.random.randint(len(bin_population[0][0]))
                direction = np.random.randint(2)
                if direction:  # faces
                    face_1 = bin_population[pair][first_parent_indexes_gen[gen_index]][:number_of_genes]
                    face_2 = bin_population[pair + 1][second_parent_indexes_gen[gen_index]][:number_of_genes]
                    bin_population[pair][first_parent_indexes_gen[gen_index]] = \
                        face_2 + bin_population[pair][first_parent_indexes_gen[gen_index]][number_of_genes:]
                    bin_population[pair + 1][second_parent_indexes_gen[gen_index]] = \
                        face_1 + bin_population[pair + 1][second_parent_indexes_gen[gen_index]][number_of_genes:]
                else:  # back
                    back_1 = bin_population[pair][first_parent_indexes_gen[gen_index]][
                             len(bin_population[pair][first_parent_indexes_gen[gen_index]]) - number_of_genes:]
                    back_2 = bin_population[pair + 1][second_parent_indexes_gen[gen_index]][
                             len(bin_population[pair][second_parent_indexes_gen[gen_index]]) - number_of_genes:]
                    bin_population[pair][first_parent_indexes_gen[gen_index]] = \
                        bin_population[pair][first_parent_indexes_gen[gen_index]][
                        :len(bin_population[pair][first_parent_indexes_gen[gen_index]]) - number_of_genes] + back_2
                    bin_population[pair + 1][second_parent_indexes_gen[gen_index]] = \
                        bin_population[pair + 1][second_parent_indexes_gen[gen_index]][
                        :len(bin_population[pair][second_parent_indexes_gen[gen_index]]) - number_of_genes] + back_1
    return bin_population

def mutation(bin_population):
    for individual in range(len(bin_population)):
        for gene in range(len(bin_population[individual])):
            if np.random.random() < MUTATION_PROBABILITY:
                number_of_gen = np.random.randint(len(bin_population[individual][gene]))
                bin_population[individual][gene] = bin_population[individual][gene][:number_of_gen] + \
                                    str((int(bin_population[individual][gene][number_of_gen]) + 1) % 2) + \
                                    bin_population[individual][gene][number_of_gen + 1:]
    return bin_population

def plotting_generation(bin_population, title):
    x_values = list(map(get_x_value_by_bin, bin_population))
    f1 = [get_value_f1(i[0], i[1]) for i in x_values]
    f2 = [get_value_f2(i[0], i[1]) for i in x_values]
    f3 = [get_value_f3(i[0], i[1]) for i in x_values]
    print(title)
    for counter, f_value in enumerate([f1, f2, f3], start=1):
        print(f'Minimum point on f{counter}: f{counter}({x_values[f_value.index(min(f_value))][0]},'
              f'{x_values[f_value.index(min(f_value))][1]}) = {min(f_value)}')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel('f1(x1,x2)')
    ax.set_ylabel('f2(x1,x2)')
    ax.set_zlabel('f3(x1,x2)')
    ax.scatter(f1, f2, f3)
    plt.show()


if __name__ == '__main__':
    ACCURACY = 2
    POPULATION_SIZE = 1200
    NUMBER_OF_GENERATIONS = 2000
    MUTATION_PROBABILITY  = 0.1
    CROSSING_OVER_PROBABILITY = 0.5
    FUNCTION_DEFINITION_LENGTH = 8 #[-4; 4]
    GENS_NUMBER = 2
    CHROMOSOMES_NUMBER = (int(np.ceil(np.log2(FUNCTION_DEFINITION_LENGTH * 10 ** ACCURACY))))
    INTERVAL = 8/ 2 ** CHROMOSOMES_NUMBER
    TITLES = ['First generation','Middle generation', 'Latest generation']


    counter_titles = 0
    population = get_initial_generation()
    for generation in range(NUMBER_OF_GENERATIONS):
        norm_vectors = fitness(population)
        population = reproduction(norm_vectors, population)
        population = crossing_over(population)
        population = mutation(population)
        if generation in [0, int(NUMBER_OF_GENERATIONS/2), NUMBER_OF_GENERATIONS-1]:
            plotting_generation(population, TITLES[counter_titles])
            counter_titles+=1
