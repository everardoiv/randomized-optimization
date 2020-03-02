import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
import seaborn as sns

from mlrose_hiive import TSPOpt
from mlrose_hiive.generators import TSPGenerator, FlipFlopGenerator, KnapsackGenerator, ContinuousPeaksGenerator
from mlrose_hiive.runners import GARunner, MIMICRunner, RHCRunner, SARunner


def mark_inflection_points(x):
    if x['Time'] < x['Next Time']:
        return 1
    else:
        return 0


def generate_graphs(run_stats, run_curves, data_strings):
    print(run_stats)

    # first figure out the rows that split
    run_curves["Inflections"] = 0
    run_curves["Next Time"] = run_curves["Time"].shift(periods=1)
    run_curves['Inflections'] = run_curves.apply(mark_inflection_points, axis=1)

    inflection_points = run_curves.index[run_curves['Inflections'] == 1].tolist()
    inflection_points.insert(0, 0)
    print("Inflection Points: ")
    print(inflection_points)
    curves = []

    for i in range(len(inflection_points) - 1):
        curves.append(pd.DataFrame(run_curves.iloc[inflection_points[i]: inflection_points[i + 1]],
                                   columns=run_curves.columns))

    curves.append(pd.DataFrame(run_curves.iloc[inflection_points[-1]:],
                               columns=run_curves.columns))

    plt.figure(dpi=100, figsize=(11, 8))
    for curve in curves:
        label = ""
        for attribute in data_strings['Parameters']:
            label += attribute + ": " + str(curve[attribute].iloc[0]) + ", "
        plt.plot(curve['Time'], curve['Fitness'], label=label)

    # plt.xlim(0, 1)
    plt.xlabel("Time (s)")
    plt.ylabel("Population Fitness")
    plt.legend(loc='best')
    plt.savefig('./Figures/' + data_strings['title'] + '_Time_vs_Fitness.png')
    plt.close()

    plt.figure(dpi=100, figsize=(11, 8))
    for curve in curves:
        label = ""
        for attribute in data_strings['Parameters']:
            print("Attribute: " + attribute)
            label += attribute + ": " + str(curve[attribute].iloc[0]) + ", "
        plt.plot(range(len(curve['Fitness'])), curve['Fitness'], label=label)

    # plt.xlim(0, 800)
    plt.xlabel("Algorithm Iterations")
    plt.ylabel("Population Fitness")
    plt.legend()
    plt.savefig('./Figures/' + data_strings['title'] + '_Iteration_vs_Fitness.png')
    plt.close()


def generateTSP(seed, number_of_cities, area_width=250, area_height=250):
    np.random.seed(seed)
    x_coords = np.random.randint(area_width, size=number_of_cities)
    y_coords = np.random.randint(area_height, size=number_of_cities)

    coords = list(tuple(zip(x_coords, y_coords)))
    duplicates = TSPGenerator.list_duplicates_(coords)

    while len(duplicates) > 0:
        for d in duplicates:
            x_coords = np.random.randint(area_width, size=len(d))
            y_coords = np.random.randint(area_height, size=len(d))
            for i in range(len(d)):
                coords[d[i]] = (x_coords[i], y_coords[i])
                pass
        duplicates = TSPGenerator.list_duplicates_(coords)
    distances = TSPGenerator.get_distances(coords, False)

    return TSPOpt(coords=coords, distances=distances, maximize=True)


def run_traveling_salesman():
    OUTPUT_DIRECTORY = './output/TSP_results/3'
    SEED = 903387974
    # Traveling Salesman
    problem = generateTSP(seed=SEED, number_of_cities=22)
    population = [350, 500]
    prob_rates = [0.01, 0.1, 0.3, 0.5]

    # Genetic Algorithm
    ga = GARunner(problem=problem,
                  experiment_name="TSP_Maximize_GA_3",
                  seed=SEED,
                  iteration_list=2 ** np.arange(8),
                  max_attempts=1000,
                  population_sizes=population,
                  mutation_rates=prob_rates,
                  output_directory=OUTPUT_DIRECTORY)
    df_ga_run_stats, df_ga_run_curves = ga.run()
    ga_data_strings = {
        'title': 'Genetic_Algorithms_22_Cities_3',
        'Parameters': ['Mutation Rate', 'Population Size']
    }
    best_fitness = max(df_ga_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_ga_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_ga_run_stats['Mutation Rate'][i])
        print(df_ga_run_stats['Population Size'][i])
        print(df_ga_run_stats['Time'][i])
        print(df_ga_run_stats['Time'][i].sum()/len(df_ga_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_ga_run_curves).set_title('GA 22 Cities ' + str(best_fitness) + ' ' + str(avg_time))
    fig.savefig('GA_22_Cities_Seaborn_3.png')
    generate_graphs(df_ga_run_stats, df_ga_run_curves, ga_data_strings)
    plt.show()

    OUTPUT_DIRECTORY = './output/TSP_results/2'
    # MIMIC Algorithm
    mmc = MIMICRunner(problem=problem,
                      experiment_name="TSP_Maximize_MIMIC_2",
                      seed=69,
                      iteration_list=2 ** np.arange(8),
                      max_attempts=500,
                      population_sizes=population,
                      keep_percent_list=prob_rates,
                      use_fast_mimic=False,
                      output_directory=OUTPUT_DIRECTORY)
    # Proportion of samples to keep at each iteration
    df_mmc_run_stats, df_mmc_run_curves = mmc.run()
    mmc_data_strings = {
        'title': 'MIMIC_22_Cities_2',
        'Parameters': ['Population Size', 'Keep Percent']
    }
    print("=============")
    best_fitness = max(df_mmc_run_stats['Fitness'])
    print(best_fitness)
    index_array = np.where(df_mmc_run_stats['Fitness'] == best_fitness)
    print("Index:")
    print(index_array)
    time = 0
    for i in index_array:
        print(df_mmc_run_stats['Keep Percent'][i])
        print(df_mmc_run_stats['Population Size'][i])
        print(df_mmc_run_stats['Time'][i])
        print(df_mmc_run_stats['Time'][i].sum()/len(df_mmc_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_mmc_run_curves).set_title('MIMIC 22 Cities ' + str(best_fitness) + ' ' + str(avg_time))
    fig.savefig('MIMIC_22_Cities_Seaborn_2.png')
    generate_graphs(df_mmc_run_stats, df_mmc_run_curves, mmc_data_strings)
    plt.show()

    # Randomized Hill Climbing
    rhc = RHCRunner(problem=problem,
                    experiment_name="TSP_Maximize_RHC_2",
                    seed=SEED,
                    iteration_list=2 ** np.arange(8),
                    max_attempts=500,
                    restart_list=[5, 15, 25, 50, 75, 95],
                    output_directory=OUTPUT_DIRECTORY)
    # Number of random restarts
    df_rhc_run_stats, df_rhc_run_curves = rhc.run()
    rhc_data_string = {
        'title': 'RHC_22_Cities_2',
        'Parameters': ['Restarts']
    }
    best_fitness = max(df_rhc_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_rhc_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_rhc_run_stats['Restarts'][i])
        print(df_rhc_run_stats['Time'][i])
        print(df_rhc_run_stats['Time'][i].sum()/len(df_rhc_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_rhc_run_curves).set_title('RHC 22 Cities ' + str(best_fitness) + ' ' + str(avg_time))
    fig.savefig('RHC_22_Cities_Seaborn_2.png')
    generate_graphs(df_rhc_run_stats, df_rhc_run_curves, rhc_data_string)
    plt.show()

    # Simulated Annealing
    sa = SARunner(problem=problem,
                  experiment_name="TSP_Maximize_SA_2",
                  seed=SEED,
                  iteration_list=2 ** np.arange(10),
                  max_attempts=1000,
                  temperature_list=[0.001, 0.05, 0.1, 0.25, 0.5, 0.8],
                  output_directory=OUTPUT_DIRECTORY)
    # Temperature is just the decay

    df_sa_run_stats, df_sa_run_curves = sa.run()
    sa_data_string = {
        'title': 'SA_22_Cities_2',
        'Parameters': ['Temperature']
    }
    best_fitness = max(df_sa_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_sa_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_sa_run_stats['Temperature'][i])
        print(df_sa_run_stats['Time'][i])
        print(df_sa_run_stats['Time'][i].sum() / len(df_sa_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_sa_run_curves).set_title('SA 22 Cities ' + str(best_fitness) + ' ' + str(avg_time))
    fig.savefig('SA_22_Cities_Seaborn_2.png')
    generate_graphs(df_sa_run_stats, df_sa_run_curves, sa_data_string)
    plt.show()


def run_flip_flop():
    # FlipFlop
    OUTPUT_DIRECTORY = './output/FlipFlop_results/3'
    SEED = 903387974
    problem = FlipFlopGenerator.generate(seed=SEED)
    population = [20, 200, 400]
    prob_rates = [0.01, 0.3, 0.5, 0.8]

    # Genetic Algorithm
    ga = GARunner(problem=problem,
                  experiment_name="FlipFlop_Maximize_GA_3",
                  seed=SEED,
                  iteration_list=2 ** np.arange(12),
                  max_attempts=500,
                  population_sizes=population,
                  mutation_rates=prob_rates,
                  output_directory=OUTPUT_DIRECTORY)
    df_ga_run_stats, df_ga_run_curves = ga.run()
    ga_data_strings = {
        'title': 'Genetic_Algorithm_FlipFlop_3',
        'Parameters': ['Mutation Rate', 'Population Size']
    }
    best_fitness = max(df_ga_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_ga_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_ga_run_stats['Mutation Rate'][i])
        print(df_ga_run_stats['Population Size'][i])
        print(df_ga_run_stats['Time'][i])
        print(df_ga_run_stats['Time'][i].sum() / len(df_ga_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_ga_run_curves).set_title("GA FlipFlop " + str(best_fitness))
    fig.savefig('GA_FlipFlop_Seaborn_3.png')
    generate_graphs(df_ga_run_stats, df_ga_run_curves, ga_data_strings)
    plt.show()

    # MIMIC Algorithm
    mmc = MIMICRunner(problem=problem,
                      experiment_name="FlipFlop_Maximize_MIMIC_3",
                      seed=SEED,
                      iteration_list=2 ** np.arange(12),
                      max_attempts=500,
                      population_sizes=population,
                      keep_percent_list=prob_rates,
                      use_fast_mimic=False,
                      output_directory=OUTPUT_DIRECTORY)
    # Proportion of samples to keep at each iteration
    df_mmc_run_stats, df_mmc_run_curves = mmc.run()
    mmc_data_strings = {
        'title': 'MIMIC_FlipFlop_3',
        'Parameters': ['Population Size', 'Keep Percent']
    }
    best_fitness = max(df_mmc_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_mmc_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_mmc_run_stats['Keep Percent'][i])
        print(df_mmc_run_stats['Population Size'][i])
        print(df_mmc_run_stats['Time'][i])
        print(df_mmc_run_stats['Time'][i].sum() / len(df_mmc_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_mmc_run_curves).set_title('MIMIC FlipFlop ' + str(best_fitness))
    fig.savefig('MIMIC_FlipFlop_Seaborn_3.png')
    generate_graphs(df_mmc_run_stats, df_mmc_run_curves, mmc_data_strings)
    plt.show()

    # Randomized Hill Climbing
    rhc = RHCRunner(problem=problem,
                    experiment_name="FlipFlop_Maximize_RHC_3",
                    seed=SEED,
                    iteration_list=2 ** np.arange(12),
                    max_attempts=500,
                    restart_list=[5, 15, 50, 75, 95],
                    output_directory=OUTPUT_DIRECTORY)
    # Number of random restarts
    df_rhc_run_stats, df_rhc_run_curves = rhc.run()
    rhc_data_string = {
        'title': 'RHC_FlipFlop_3',
        'Parameters': ['Restarts']
    }
    best_fitness = max(df_rhc_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_rhc_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_rhc_run_stats['Restarts'][i])
        print(df_rhc_run_stats['Time'][i])
        print(df_rhc_run_stats['Time'][i].sum()/len(df_rhc_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_rhc_run_curves).set_title('RHC FlipFlop ' + str(best_fitness))
    fig.savefig('RHC_FlipFlop_Seaborn_3.png')
    generate_graphs(df_rhc_run_stats, df_rhc_run_curves, rhc_data_string)
    plt.show()

    OUTPUT_DIRECTORY = './output/FlipFlop_results/2'
    # Simulated Annealing
    sa = SARunner(problem=problem,
                  experiment_name="FlipFlop_Maximize_SA_2",
                  seed=SEED,
                  iteration_list=2 ** np.arange(12),
                  max_attempts=1000,
                  temperature_list=[0.001, 0.05, 0.1, 0.25, 0.5, 0.8],
                  output_directory=OUTPUT_DIRECTORY)
    # Temperature is just the decay

    df_sa_run_stats, df_sa_run_curves = sa.run()
    sa_data_string = {
        'title': 'SA_FlipFlop_2',
        'Parameters': ['Temperature']
    }
    best_fitness = max(df_sa_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_sa_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_sa_run_stats['Temperature'][i])
        print(df_sa_run_stats['Time'][i])
        print(df_sa_run_stats['Time'][i].sum()/len(df_sa_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_sa_run_curves).set_title('SA FlipFlop ' + str(best_fitness))
    fig.savefig('SA_FlipFlop_Seaborn_2.png')
    generate_graphs(df_sa_run_stats, df_sa_run_curves, sa_data_string)
    plt.show()


def run_knapsack():
    # Knapsack
    OUTPUT_DIRECTORY = './output/Knapsack_results/4'
    SEED = 903387974
    problem = KnapsackGenerator.generate(seed=SEED, number_of_items_types=12, max_item_count=6, max_weight_per_item=26,
                                         max_value_per_item=12, max_weight_pct=0.8)
    population = [50, 130, 210, 500]
    prob_rates = [0.05, 0.3, 0.5, 0.8]

    # Genetic Algorithm
    ga = GARunner(problem=problem,
                  experiment_name="Knapsack_Maximize_GA_4",
                  seed=SEED,
                  iteration_list=2 ** np.arange(6),
                  max_attempts=500,
                  population_sizes=population,
                  mutation_rates=prob_rates,
                  output_directory=OUTPUT_DIRECTORY)
    df_ga_run_stats, df_ga_run_curves = ga.run()
    ga_data_strings = {
        'title': 'Genetic_Algorithms_Knapsack_4',
        'Parameters': ['Mutation Rate', 'Population Size']
    }
    best_fitness = max(df_ga_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_ga_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_ga_run_stats['Mutation Rate'][i])
        print(df_ga_run_stats['Population Size'][i])
        print(df_ga_run_stats['Time'][i])
        print(df_ga_run_stats['Time'][i].sum()/len(df_ga_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")

    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_ga_run_curves).set_title('GA_Knapsack_' + str(best_fitness) + '_' + str(avg_time))
    fig.savefig('GA_Knapsack_Seaborn_4.png')
    generate_graphs(df_ga_run_stats, df_ga_run_curves, ga_data_strings)
    plt.show()

    # MIMIC Algorithm
    mmc = MIMICRunner(problem=problem,
                      experiment_name="Knapsack_Maximize_MIMIC_4",
                      seed=SEED,
                      iteration_list=2 ** np.arange(6),
                      max_attempts=500,
                      population_sizes=population,
                      keep_percent_list=prob_rates,
                      use_fast_mimic=False,
                      output_directory=OUTPUT_DIRECTORY)
    # Proportion of samples to keep at each iteration
    df_mmc_run_stats, df_mmc_run_curves = mmc.run()
    mmc_data_strings = {
        'title': 'MIMIC_Knapsack_4',
        'Parameters': ['Population Size', 'Keep Percent']
    }
    best_fitness = max(df_mmc_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_mmc_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_mmc_run_stats['Keep Percent'][i])
        print(df_mmc_run_stats['Population Size'][i])
        print(df_mmc_run_stats['Time'][i])
        print(df_mmc_run_stats['Time'][i].sum()/len(df_mmc_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_mmc_run_curves).set_title('MIMIC_Knapsack_' + str(best_fitness) + '_' + str(avg_time))
    fig.savefig('MIMIC_Knapsack_Seaborn_4.png')
    generate_graphs(df_mmc_run_stats, df_mmc_run_curves, mmc_data_strings)
    plt.show()

    # Randomized Hill Climbing
    rhc = RHCRunner(problem=problem,
                    experiment_name="Knapsack_Maximize_RHC_4",
                    seed=SEED,
                    iteration_list=2 ** np.arange(2),
                    max_attempts=250,
                    restart_list=[5, 15, 50],
                    output_directory=OUTPUT_DIRECTORY)
    # Number of random restarts
    df_rhc_run_stats, df_rhc_run_curves = rhc.run()
    rhc_data_string = {
        'title': 'RHC_Knapsack_4',
        'Parameters': ['Restarts']
    }
    best_fitness = max(df_rhc_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_rhc_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_rhc_run_stats['Restarts'][i])
        print(df_rhc_run_stats['Time'][i])
        print(df_rhc_run_stats['Time'][i].sum()/len(df_rhc_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_rhc_run_curves).set_title('RHC_Knapsack_' + str(best_fitness) + '_' + str(avg_time))
    fig.savefig('RHC_Knapsack_Seaborn_4.png')
    generate_graphs(df_rhc_run_stats, df_rhc_run_curves, rhc_data_string)
    plt.show()

    # Simulated Annealing
    sa = SARunner(problem=problem,
                  experiment_name="Knapsack_Maximize_SA_4",
                  seed=SEED,
                  iteration_list=2 ** np.arange(8),
                  max_attempts=1000,
                  temperature_list=[0.001, 0.05, 0.1, 0.25, 0.5, 0.8],
                  output_directory=OUTPUT_DIRECTORY)
    # Temperature is just the Geometric decay

    df_sa_run_stats, df_sa_run_curves = sa.run()
    sa_data_string = {
        'title': 'SA_Knapsack_4',
        'Parameters': ['Temperature']
    }
    best_fitness = max(df_sa_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_sa_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_sa_run_stats['Temperature'][i])
        print(df_sa_run_stats['Time'][i])
        print(df_sa_run_stats['Time'][i].sum()/len(df_sa_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_sa_run_curves).set_title('SA_Knapsack_' + str(best_fitness) + '_' + str(avg_time))
    fig.savefig('SA_Knapsack_Seaborn_4.png')
    generate_graphs(df_sa_run_stats, df_sa_run_curves, sa_data_string)
    plt.show()


def run_sixpeaks():
    # Six Peaks
    OUTPUT_DIRECTORY = './output/SixPeaks_results/2'
    SEED = 903387974
    problem = ContinuousPeaksGenerator.generate(seed=SEED, t_pct=0.2)
    population = [15, 25, 50, 80, 130, 210]
    prob_rates = [0.05, 0.1, 0.3, 0.5, 0.8]

    # Genetic Algorithm
    ga = GARunner(problem=problem,
                  experiment_name="SixPeaks_Maximize_GA_2",
                  seed=SEED,
                  iteration_list=2 ** np.arange(8),
                  max_attempts=500,
                  population_sizes=population,
                  mutation_rates=prob_rates,
                  output_directory=OUTPUT_DIRECTORY)
    df_ga_run_stats, df_ga_run_curves = ga.run()
    ga_data_strings = {
        'title': 'Genetic_Algorithms_SixPeaks_2',
        'Parameters': ['Mutation Rate', 'Population Size']
    }
    best_fitness = max(df_ga_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_ga_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_ga_run_stats['Mutation Rate'][i])
        print(df_ga_run_stats['Population Size'][i])
        print(df_ga_run_stats['Time'][i])
        print(df_ga_run_stats['Time'][i].sum()/len(df_ga_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_ga_run_curves).set_title('GA SixPeaks ' + str(best_fitness))
    fig.savefig('GA_SixPeaks_Seaborn_2.png')
    generate_graphs(df_ga_run_stats, df_ga_run_curves, ga_data_strings)
    plt.show()

    # MIMIC Algorithm
    mmc = MIMICRunner(problem=problem,
                      experiment_name="SixPeaks_Maximize_MIMIC_2",
                      seed=SEED,
                      iteration_list=2 ** np.arange(10),
                      max_attempts=500,
                      population_sizes=population,
                      keep_percent_list=prob_rates,
                      use_fast_mimic=False,
                      output_directory=OUTPUT_DIRECTORY)
    # Proportion of samples to keep at each iteration
    df_mmc_run_stats, df_mmc_run_curves = mmc.run()
    mmc_data_strings = {
        'title': 'MIMIC_SixPeaks_2',
        'Parameters': ['Population Size', 'Keep Percent']
    }
    best_fitness = max(df_mmc_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_mmc_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_mmc_run_stats['Keep Percent'][i])
        print(df_mmc_run_stats['Population Size'][i])
        print(df_mmc_run_stats['Time'][i])
        print(df_mmc_run_stats['Time'][i].sum() / len(df_mmc_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_mmc_run_curves).set_title('MIMIC SixPeaks ' + str(best_fitness))
    fig.savefig('MIMIC_SixPeaks_Seaborn_2.png')
    generate_graphs(df_mmc_run_stats, df_mmc_run_curves, mmc_data_strings)
    plt.show()

    # Randomized Hill Climbing
    rhc = RHCRunner(problem=problem,
                    experiment_name="SixPeaks_Maximize_RHC_2",
                    seed=SEED,
                    iteration_list=2 ** np.arange(10),
                    max_attempts=1000,
                    restart_list=[5, 15, 25, 50, 75, 95, 100],
                    output_directory=OUTPUT_DIRECTORY)
    # Number of random restarts
    df_rhc_run_stats, df_rhc_run_curves = rhc.run()
    rhc_data_string = {
        'title': 'RHC_SixPeaks_2',
        'Parameters': ['Restarts']
    }
    best_fitness = max(df_rhc_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_rhc_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_rhc_run_stats['Restarts'][i])
        print(df_rhc_run_stats['Time'][i])
        print(df_rhc_run_stats['Time'][i].sum() / len(df_rhc_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_rhc_run_curves).set_title('RHC SixPeaks ' + str(best_fitness))
    fig.savefig('RHC_SixPeaks_Seaborn_2.png')
    generate_graphs(df_rhc_run_stats, df_rhc_run_curves, rhc_data_string)
    plt.show()

    # Simulated Annealing
    sa = SARunner(problem=problem,
                  experiment_name="SixPeaks_Maximize_SA_2",
                  seed=SEED,
                  iteration_list=2 ** np.arange(10),
                  max_attempts=1000,
                  temperature_list=[0.001, 0.05, 0.1, 0.25, 0.5, 0.8],
                  output_directory=OUTPUT_DIRECTORY)
    # Temperature is just the Geometric decay

    df_sa_run_stats, df_sa_run_curves = sa.run()
    sa_data_string = {
        'title': 'SA_SixPeaks_2',
        'Parameters': ['Temperature']
    }
    best_fitness = max(df_sa_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_sa_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_sa_run_stats['Temperature'][i])
        print(df_sa_run_stats['Time'][i])
        print(df_sa_run_stats['Time'][i].sum() / len(df_sa_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_sa_run_curves).set_title('SA SixPeaks ' + str(best_fitness))
    fig.savefig('SA_SixPeaks_Seaborn_2.png')
    generate_graphs(df_sa_run_stats, df_sa_run_curves, sa_data_string)
    plt.show()


if __name__ == "__main__":
    run_traveling_salesman()
    # run_flip_flop()
    # run_knapsack()
    # run_sixpeaks()
