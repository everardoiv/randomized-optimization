import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlrose_hiive as mlrose
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

from mlrose_hiive.runners import NNGSRunner, SKMLPRunner


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


def plot_fitness(name, values, param1, param2, param3):
    df = pd.DataFrame(values, columns=[param1, param2, param3])
    df = df.pivot(index=param1, columns=param2, values=param3) \
        .reset_index()
    df = df.melt(param1, var_name=param2, value_name=param3)
    g = sns.catplot(x=param1, y=param3, hue=param2, data=df)
    title = f"Plot for a %s" % "\n".join(wrap(f"{name}", width=150))
    g.fig.suptitle(title)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig('./graphs/' + name.replace(" ", "_") + '_' + str(timestr) + '.png')
    plt.close()


# code copied from https://www.dataquest.io/blog/learning-curves-machine-learning/
def learning_curves(name, estimator, data, features, target, train_sizes, cv):

    train_sizes, train_scores, validation_scores = learning_curve(estimator, data[features], data[target],
                                                                 train_sizes=train_sizes, cv=cv,
                                                                 scoring ='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    # title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    title = f"Learning curves for a\n%s" % "\n".join(wrap(f"{name}", width=60))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    plt.title(title, fontsize=14, y=1.03)
    plt.margins(0)
    plt.legend()
    # plt.show()
    plt.savefig('./learning_curves/' + name + '_' + str(timestr) + '.png')
    plt.close()


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring='neg_log_loss', loss=True):
    """
    Credit to sklearn tutorial on plotting learning curves:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    scoring: scoring metric to evaluate against
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    train_labels = {'f1_macro': "Macro F1", 'neg_log_loss': "Cross-Entropy Loss"}

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel(train_labels[scoring])

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, scoring=scoring, random_state=1, verbose=10)
    train_scores_mean = np.mean(train_scores, axis=1) * (1.0 + (-2 * loss))
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1) * (1.0 + (-2 * loss))
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="darkorange")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="navy")
    axes[0].plot(train_sizes, train_scores_mean, 'x-', color="darkorange",
                 label="training")
    axes[0].plot(train_sizes, test_scores_mean, 'x-', color="navy",
                 label="validation")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'x-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit times")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'x-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Fit times")
    axes[2].set_ylabel(train_labels[scoring])

    return train_sizes, fit_times, plt


def split_the_data(df, column, cv, stratify_bool):
    X = df.drop(columns=column, axis=1)
    y = df[column]

    # Stratify makes it so that the proportion of values in the sample in our test group will be the same as the
    # proportion of values provided to parameter stratify
    if stratify_bool:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=42)

    return X, y, X_train, X_test, y_train, y_test


def load_and_describe_data(file):
    data = pd.read_csv(file)

    if file == 'data/output/heart-disease-data.csv':
        data['ca'] = pd.to_numeric(data['ca'], errors='coerce').interpolate()
        data['thal'] = pd.to_numeric(data['thal'], errors='coerce').interpolate()

    assert not data.isnull().values.any()

    # Frequency Statistics - run once
    # csv_file = file.split('/')
    #
    # describe_file = 'data/statistics/description-' + csv_file[2]
    # data.describe(include='all').to_csv(describe_file, header=True)
    # # Less than or greater than -1/+1 is highly skewed
    # skew_file = 'data/statistics/skew-' + csv_file[2]
    # data.skew(axis=0).to_csv(skew_file, header=True)
    #
    # plot_data(csv_file[2], data)

    return data


def run_neural_net_ga(X_train, y_train, X_test, y_test):
    # Genetic Algorithm
    # , (10, 10, 10), (10, 10, 10, 10, 10, 10)
    # , mlrose.neural.activation.tanh
    grid_search_params = ({
        'max_iters': [1, 2, 4, 8, 16, 32, 64, 128],
        'learning_rate': [0.001, 0.002, 0.003],
        'hidden_layer_sizes': [(23, 23), (100,)],
        'activation': [mlrose.neural.activation.relu],
        'pop_size': [10, 100, 350, 1000, 2000],
        'mutation_prob': [0.1, 0.3, 0.5, 0.8]
    })

    mlp_data_strings = {
        'title': 'NN-GA',
        'Parameters': ['hidden_layer_sizes', 'activation']
    }

    mlp = NNGSRunner(x_train=X_train,
                     y_train=y_train,
                     x_test=X_test,
                     y_test=y_test,
                     experiment_name='NN-GA',
                     algorithm=mlrose.algorithms.genetic_alg,
                     iteration_list=[1, 16, 128],
                     grid_search_parameters=grid_search_params,
                     generate_curves=True,
                     seed=SEED,
                     early_stopping=False,
                     clip_max=1e+10,
                     max_attempts=10,
                     bias=True,
                     n_jobs=4,
                     cv=3,
                     output_directory='./output_nn/GA')

    results = mlp.run()
    print(results[3].best_score_, results[3].best_params_)
    activate = results[3].best_params_['activation']
    hl_sizes = results[3].best_params_['hidden_layer_sizes']
    learning = results[3].best_params_['learning_rate']
    max_iter = results[3].best_params_['max_iters']
    pop_size = results[3].best_params_['pop']
    mutation = results[3].best_params_['mutation_prob']
    print(results[2])
    df_mlp_run_stats, df_mlp_run_curves = results[0], results[1]

    best_fitness = max(df_mlp_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_mlp_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_mlp_run_stats['Time'][i])
        print(df_mlp_run_stats['Time'][i].sum() / len(df_mlp_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")

    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_mlp_run_curves).set_title(
        'NN-GA ' + str(target_feature) + ' ' + str(best_fitness) + ' | ' + str(avg_time))
    fig.savefig('NN_GA.png')
    generate_graphs(df_mlp_run_stats, df_mlp_run_curves, mlp_data_strings)
    plt.show()

    nn_ga_model = mlrose.NeuralNetwork(hidden_nodes=hl_sizes,
                                       activation='relu',
                                       algorithm='genetic_alg',
                                       max_iters=max_iter,
                                       learning_rate=learning,
                                       max_attempts=10,
                                       pop_size=pop_size,
                                       mutation_prob=mutation,
                                       curve=True,
                                       random_state=SEED)
    title = 'NN-GA-2'
    ga_train_size, ga_fit_times, _ = plot_learning_curve(nn_ga_model, title, X_train, y_train, cv=3,
                                                         scoring='f1_macro', loss=False, n_jobs=-1)
    plt.show()
    nn_ga_model.fit(X_train, y_train)
    nn_ga_pred = nn_ga_model.predict(X_test)
    print(classification_report(y_test, nn_ga_pred, digits=4))


def run_neural_net_sa(X_train, y_train, X_test, y_test):
    # Simulated Annealing
    # , (10, 10, 10), (10, 10, 10, 10, 10, 10)
    # , mlrose.neural.activation.tanh

    decays = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    param_range = [mlrose.algorithms.GeomDecay(decay=decay) for decay in decays]

    grid_search_params = ({
        'max_iters': [1, 2, 4, 8, 16, 32, 64, 128],
        'learning_rate': [0.001, 0.002, 0.003],
        'hidden_layer_sizes': [(23, 23), (100,)],
        'activation': [mlrose.neural.activation.relu],
        'schedule': param_range
    })

    mlp_data_strings = {
        'title': 'NN-SA',
        'Parameters': ['hidden_layer_sizes', 'activation', 'schedule']
    }

    mlp = NNGSRunner(x_train=X_train,
                     y_train=y_train,
                     x_test=X_test,
                     y_test=y_test,
                     experiment_name='NN-SA',
                     algorithm=mlrose.algorithms.simulated_annealing,
                     iteration_list=[1, 10, 100, 1000],
                     grid_search_parameters=grid_search_params,
                     bias=True,
                     early_stopping=False,
                     clip_max=1e+10,
                     max_attempts=10,
                     generate_curves=True,
                     seed=SEED,
                     output_directory='./output_nn/SA')
    results = mlp.run()
    print(results[3].best_score_, results[3].best_params_)
    activate = results[3].best_params_['activation']
    hl_sizes = results[3].best_params_['hidden_layer_sizes']
    learning = results[3].best_params_['learning_rate']
    max_iter = results[3].best_params_['max_iters']
    schedule = results[3].best_params_['schedule']
    print(results[2])
    df_mlp_run_stats, df_mlp_run_curves = results[0], results[1]

    best_fitness = max(df_mlp_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_mlp_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_mlp_run_stats['Time'][i])
        print(df_mlp_run_stats['Time'][i].sum() / len(df_mlp_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_mlp_run_curves).set_title(
        'NN-SA ' + str(target_feature) + ' ' + str(best_fitness) + ' | ' + str(avg_time))
    fig.savefig('NN_SA.png')
    generate_graphs(df_mlp_run_stats, df_mlp_run_curves, mlp_data_strings)
    plt.show()

    nn_sa_model = mlrose.NeuralNetwork(hidden_nodes=hl_sizes,
                                       activation='relu',
                                       algorithm='simulated_annealing',
                                       max_iters=max_iter,
                                       learning_rate=learning,
                                       max_attempts=10,
                                       schedule=schedule,
                                       curve=True,
                                       random_state=SEED)
    title = 'NN-SA-2'
    sa_train_size, sa_fit_times, _ = plot_learning_curve(nn_sa_model, title, X_train, y_train, cv=3,
                                                         scoring='f1_macro', loss=False, n_jobs=-1)
    plt.show()
    nn_sa_model.fit(X_train, y_train)
    nn_sa_pred = nn_sa_model.predict(X_test)
    print(classification_report(y_test, nn_sa_pred, digits=4))


def run_neural_net_rhc(X_train, y_train, X_test, y_test):
    # RHC
    # , (10, 10, 10), (10, 10, 10, 10, 10, 10)
    # , mlrose.neural.activation.tanh
    grid_search_params = ({
        'max_iters': [1, 2, 4, 8, 16, 32, 64, 128],
        'learning_rate': [0.001, 0.002, 0.003],
        'hidden_layer_sizes': [(23, 23), (100,)],
        'activation': [mlrose.neural.activation.relu]
    })

    mlp_data_strings = {
        'title': 'NN-RHC',
        'Parameters': ['hidden_layer_sizes', 'activation']
    }

    mlp = NNGSRunner(x_train=X_train,
                     y_train=y_train,
                     x_test=X_test,
                     y_test=y_test,
                     experiment_name='NN-RHC',
                     algorithm=mlrose.algorithms.random_hill_climb,
                     iteration_list=[1, 10, 100, 1000],
                     grid_search_parameters=grid_search_params,
                     bias=True,
                     early_stopping=False,
                     clip_max=1e+10,
                     max_attempts=10,
                     generate_curves=True,
                     seed=SEED,
                     output_directory='./output_nn/RHC')

    results = mlp.run()
    print(results[3].best_score_, results[3].best_params_)
    activate = results[3].best_params_['activation']
    hl_sizes = results[3].best_params_['hidden_layer_sizes']
    learning = results[3].best_params_['learning_rate']
    max_iter = results[3].best_params_['max_iters']
    print(results[2])
    df_mlp_run_stats, df_mlp_run_curves = results[0], results[1]

    best_fitness = max(df_mlp_run_stats['Fitness'])
    print("=============")
    print(best_fitness)
    index_array = np.where(df_mlp_run_stats['Fitness'] == best_fitness)
    time = 0
    for i in index_array:
        print(df_mlp_run_stats['Time'][i])
        print(df_mlp_run_stats['Time'][i].sum() / len(df_mlp_run_stats['Time'][i]))
    avg_time = time / len(index_array)
    print(avg_time)
    print("============")
    fig = plt.figure(dpi=100, figsize=(11, 8))
    sns.lineplot(x='Iteration', y='Fitness', data=df_mlp_run_curves).set_title(
        'NN-RHC ' + str(target_feature) + ' ' + str(best_fitness) + ' | ' + str(avg_time))
    fig.savefig('NN_RHC.png')
    generate_graphs(df_mlp_run_stats, df_mlp_run_curves, mlp_data_strings)
    plt.show()

    nn_rhc_model = mlrose.NeuralNetwork(hidden_nodes=hl_sizes,
                                        activation='relu',
                                        algorithm='random_hill_climb',
                                        max_iters=max_iter,
                                        learning_rate=learning,
                                        max_attempts=10,
                                        curve=True,
                                        random_state=SEED)
    title = 'NN-RHC-2'
    rhc_train_size, rhc_fit_times, _ = plot_learning_curve(nn_rhc_model, title, X_train, y_train, cv=3,
                                                           scoring='f1_macro', loss=False, n_jobs=-1)
    plt.show()
    nn_rhc_model.fit(X_train, y_train)
    nn_rhc_pred = nn_rhc_model.predict(X_test)
    print(classification_report(y_test, nn_rhc_pred, digits=4))


def run_sklearn_mlp(X_train, y_train, X_test, y_test):
    # Backpropagation
    # (10, 10, 10), (10, 10, 10, 10, 10, 10)
    # , mlrose.neural.activation.tanh
    grid_search_params = ({
        'max_iters': [1, 2, 4, 8, 16, 32, 64, 128],
        'hidden_layer_sizes': [(23, 23), (100,)],
        'activation': [mlrose.neural.activation.relu]
    })

    mlp_data_strings = {
        'title': 'SK-MLP',
        'Parameters': ['hidden_layer_sizes', 'activation']
    }

    mlp = SKMLPRunner(x_train=X_train,
                     y_train=y_train,
                     x_test=X_test,
                     y_test=y_test,
                     experiment_name='SKLearn MLP',
                     iteration_list=[1, 10, 100, 1000],
                     grid_search_parameters=grid_search_params,
                     n_jobs=4,
                     cv=3,
                     early_stopping=False,
                     max_attempts=10,
                     generate_curves=True,
                     seed=SEED,
                     output_directory='./output_nn/SK_MLP')

    results = mlp.run()
    print(results[3].best_score_, results[3].best_params_)
    activate = results[3].best_params_['activation']
    hl_sizes = results[3].best_params_['hidden_layer_sizes']
    # learning = results[3].best_params_['learning_rate']
    # max_iter = results[3].best_params_['max_iters']
    print(results[2])
    # df_mlp_run_stats, df_mlp_run_curves = results[0], results[1]
    #
    # best_fitness = max(df_mlp_run_stats['Fitness'])
    # print("=============")
    # print(best_fitness)
    # index_array = np.where(df_mlp_run_stats['Fitness'] == best_fitness)
    # time = 0
    # for i in index_array:
    #     print(df_mlp_run_stats['Time'][i])
    #     print(df_mlp_run_stats['Time'][i].sum() / len(df_mlp_run_stats['Time'][i]))
    # avg_time = time / len(index_array)
    # print(avg_time)
    # print("============")
    # fig = plt.figure(dpi=100, figsize=(11, 8))
    # sns.lineplot(x='Iteration', y='Fitness', data=df_mlp_run_curves).set_title(
    #     'SKMLP ' + str(target_feature) + ' ' + str(best_fitness) + ' | ' + str(avg_time))
    # fig.savefig('SKLMLP.png')
    # generate_graphs(df_mlp_run_stats, df_mlp_run_curves, mlp_data_strings)
    # plt.show()
    nn_mlp_model = mlrose.NeuralNetwork(hidden_nodes=hl_sizes,
                                        activation='relu',
                                        algorithm='gradient_descent',
                                        max_attempts=10,
                                        curve=True,
                                        random_state=SEED)

    mlp = MLPClassifier(hidden_layer_sizes=hl_sizes,
                        activation='relu',
                        random_state=SEED)
    title = 'NN-MLP-2'
    rhc_train_size, rhc_fit_times, _ = plot_learning_curve(nn_mlp_model, title, X_train, y_train, cv=3,
                                                           scoring='f1_macro', loss=False, n_jobs=-1)
    plt.show()
    nn_mlp_model.fit(X_train, y_train)
    nn_mlp_pred = nn_mlp_model.predict(X_test)
    print(classification_report(y_test, nn_mlp_pred, digits=4))


if __name__ == '__main__':
    # Get the experiment
    experiments = ['Credit Card', 'Poker Hand']
    data_sets_filename = ['credit-card-data.csv', 'sampled-poker-hand-data.csv']
    feature_label = ['default payment next month', 'Poker Hand']
    cross_validation = [5, 3]
    model = MLPClassifier(random_state=42)
    SEED = 903387974

    for i in range(2):
        filename = data_sets_filename[i]
        cross_v = cross_validation[i]
        experiment = experiments[i]
        target_feature = feature_label[i]

        data = load_and_describe_data(filename)
        print('%s - %s - %s' % (filename, experiment, cross_v))

        trimmed_experiment = experiment.replace(" ", "")

        # TODO: Use StandardScaler on X
        x_scaler = StandardScaler()
        X = data.drop(columns=target_feature, axis=1)
        print(list(X.columns))
        data[list(X.columns)] = x_scaler.fit_transform(data.drop(columns=target_feature, axis=1))
        print(data)
        X, y, X_train, X_test, y_train, y_test = split_the_data(data, target_feature, cross_v, True)

        # hidden_node: List giving the number of nodes in each hidden layer
        # activation: 'identity', 'relu', 'sigmoid', 'tanh'
        # algorithm: 'random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent'
        # max_iters: 100
        # bias: True
        # is_classifier: True
        # learning_rate: 0.1
        # early_stopping: False
        # clip_max: limit weights to a range
        # restarts: Number of random restarts
        # schedule: mlrose_hiive.GeomDecay()
        # pop_size: genetic_alg
        # mutation_prob: genetic_alg
        # max_attempts: genetic_alg
        # random_state:
        # curve: return fitness curve or not

        # NN Classifier is for tuning the weights
        # SKML Classifier is for backpropagation

        run_sklearn_mlp(X_train, y_train, X_test, y_test)

        run_neural_net_ga(X_train, y_train, X_test, y_test)

        run_neural_net_sa(X_train, y_train, X_test, y_test)

        run_neural_net_rhc(X_train, y_train, X_test, y_test)

        test_acc = []
        train_acc = []

        # Solve using genetic algorithm
        algorithms = ['genetic_alg', 'simulated_annealing', 'random_hill_climb']

        for algo in algorithms:

            model = mlrose.NeuralNetwork(hidden_nodes=[10, 10, 10], activation='tanh',
                                         algorithm=algo,
                                         max_iters=20000, bias=True, is_classifier=True,
                                         learning_rate=0.001, early_stopping=False,
                                         clip_max=5, max_attempts=100, random_state=SEED)

            model.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = model.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
            train_acc.append(y_train_accuracy)
            print(algo)
            print(y_train_accuracy)
            # Predict labels for test set and assess accuracy
            y_test_pred = model.predict(X_test)

            y_test_accuracy = accuracy_score(y_test, y_test_pred)

            print(y_test_accuracy)
            test_acc.append(y_test_accuracy)
            print()

            figure = plot_learning_curve(model, str('NN ' + algo), X_train, y_train)
            figure.savefig(str("NN " + algo + ".png"))

            model = mlrose.NeuralNetwork(hidden_nodes=[100,], activation='relu',
                                         algorithm=algo,
                                         max_iters=20000, bias=True, is_classifier=True,
                                         learning_rate=0.001, early_stopping=False,
                                         clip_max=5, max_attempts=100, random_state=SEED)

            model.fit(X_train, y_train)
            # Predict labels for train set and assess accuracy
            y_train_pred = model.predict(X_train)
            y_train_accuracy = accuracy_score(y_train, y_train_pred)
            train_acc.append(y_train_accuracy)
            print(algo)
            print(y_train_accuracy)
            # Predict labels for test set and assess accuracy
            y_test_pred = model.predict(X_test)

            y_test_accuracy = accuracy_score(y_test, y_test_pred)

            print(y_test_accuracy)
            test_acc.append(y_test_accuracy)
            print()
