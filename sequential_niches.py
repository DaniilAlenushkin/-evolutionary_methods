import matplotlib.pyplot as plt
import numpy as np

def get_function_value(values: list):
    result = []
    for i in values:
        value = np.sin(5 * np.pi * (i ** 0.75 - 0.05)) ** 6
        for j in stop_list:
            if j[0] < i < j[1]:
                value = 0
                break
        result.append(value)
    return result


def get_initial_generation(number_of_ind, number_of_bits_in_ind):
    return [''.join(list(map(lambda x: str(x),
                             list(np.random.randint(2, size=number_of_bits_in_ind)))))
            for _ in range(number_of_ind)]


def get_value_by_bin(population):
    dec_population = list(map(lambda x: int(x, 2), population))
    dec_population_value = get_function_value(list(map(lambda x: x * interval, dec_population)))
    return dec_population, dec_population_value


def fitness_function(bin_population):
    dec_population, dec_population_value = get_value_by_bin(bin_population)
    norm_dec_population = [(i/sum(dec_population_value)) for i in dec_population_value]
    return norm_dec_population


def reproduction(vector, bin_population):
    ruler = []
    counter = 0
    for i in vector:
        ruler.append([counter, counter + i])
        counter += i
    spin = np.random.random(size=100)
    result = []
    for value in spin:
        for step in range(len(ruler)):
            if ruler[step][0] <= value <= ruler[step][1]:
                result.append(bin_population[step])
                break
    return result


def crossing_over(bin_population):
    np.random.shuffle(bin_population)
    for i in range(0, len(bin_population), 2):
        if np.random.randint(2):
            number_of_genes = np.random.randint(len(bin_population[0]))
            direction = np.random.randint(2)
            if direction:  # faces
                face_1 = bin_population[i][:number_of_genes]
                face_2 = bin_population[i + 1][:number_of_genes]
                bin_population[i] = face_2 + bin_population[i][number_of_genes:]
                bin_population[i + 1] = face_1 + bin_population[i + 1][number_of_genes:]
            else:  # back
                back_1 = bin_population[i][len(bin_population[i]) - number_of_genes:]
                back_2 = bin_population[i + 1][len(bin_population[i]) - number_of_genes:]
                bin_population[i] = bin_population[i][:len(bin_population[i]) - number_of_genes] + back_2
                bin_population[i + 1] = bin_population[i + 1][:len(bin_population[i]) - number_of_genes] + back_1
    return bin_population


def mutation(bin_population):
    for i in range(len(bin_population)):
        if not np.random.randint(999):
            number_of_gen = np.random.randint(len(bin_population[i]))
            bin_population[i] = bin_population[i][:number_of_gen] + \
                                str((int(bin_population[i][number_of_gen]) + 1) % 2) + \
                                bin_population[i][number_of_gen + 1:]
    return bin_population


def plotting_generation(bin_population, x_coord, f):
    dec_population, dec_population_value = get_value_by_bin(bin_population)
    x_pop = list(map(lambda x: x * interval, dec_population))
    fig, ax = plt.subplots()
    ax.plot(x_coord, f)
    ax.grid()
    ax.scatter(x_pop, dec_population_value, s=10, c='r')
    plt.show()


def zeroing_the_peak(bin_population, f):
    dec_population, dec_population_value = get_value_by_bin(bin_population)
    x_peak = dec_population[(dec_population_value.index(max(dec_population_value)))] * interval
    for i in range(len(x_coordinate)):
        if (x_peak - 0.13) < x_coordinate[i] < (x_peak + 0.13):
            f[i] = 0
    return f, [x_peak - 0.13, x_peak + 0.13]


def get_final_graph(populations):
    fig, ax = plt.subplots()
    x_coordinate = np.linspace(0.0, 1, num=1000)
    main_function = [np.sin(5 * np.pi * (i ** 0.75 - 0.05)) ** 6 for i in x_coordinate]
    ax.plot(x_coordinate, main_function)
    for i in populations:
        dec_population, dec_population_value = get_value_by_bin(i)
        x_pop = list(map(lambda x: x * interval, dec_population))
        print(x_pop[dec_population_value.index(max(dec_population_value))], max(dec_population_value))
        ax.scatter(x_pop, dec_population_value, s=10)
    ax.grid()
    ax.legend(('main function', '1 iteration',
               '2 iteration', '3 iteration',
               '4 iteration', '5 iteration'))
    plt.show()


if __name__ == '__main__':
    perfect_populations = []
    accuracy = 3
    interval = 1 / 2 ** int(np.ceil(np.log2(10 ** accuracy)))
    x_coordinate = np.linspace(0.0, 1, num=1000)
    main_function = [np.sin(5 * np.pi * (i ** 0.75 - 0.05)) ** 6 for i in x_coordinate]
    stop_list = []
    for iteration in range(1, 6):
        main_population = get_initial_generation(200, int(np.ceil(np.log2(10 ** accuracy))))
        for i in range(iteration*500):
            norm_vector = fitness_function(main_population)
            main_population = reproduction(norm_vector, main_population)
            main_population = crossing_over(main_population)
            main_population = mutation(main_population)
            if i in [0, int(iteration*500/2), int(iteration*500)-1]:
                plotting_generation(main_population, x_coordinate, main_function)
        perfect_populations.append(main_population)
        main_function, stop_intervals = zeroing_the_peak(main_population, main_function)
        stop_list.append(stop_intervals)
    stop_list = []
    get_final_graph(perfect_populations)
