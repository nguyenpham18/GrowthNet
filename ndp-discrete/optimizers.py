import numpy as np
import psutil
import time
from utils import x0_sampling
import cma
import pandas as pd
from multiprocess import Pool


def CMAES(config, fitness):
    nb_parameters = config["nb_trainable_parameters"]
    x0 = x0_sampling(config["x0_dist"], nb_parameters)
    es = cma.CMAEvolutionStrategy(
        x0,
        config["sigma_init"],
        {
            "verb_disp": config["print_every"],
            "popsize": config["popsize"],
            "maxiter": config["generations"],
            "seed": config["seed"],
            "CMA_elitist": config["CMA_elitist"],
            "minstd": config["minstd"],
        },
    )

    print("\n.......................................................")
    print("\nInitilisating CMA-ES with", nb_parameters, "trainable parameters \n")

    print("\n ♪┏(°.°)┛┗(°.°)┓ Starting Evolution ┗(°.°)┛┏(°.°)┓ ♪ \n")
    tic = time.time()

    objective_solution_best = np.inf
    objective_solution_centroid = np.inf
    objectives_centroid = []
    objectives_best = []
    gen = 0
    early_stopping_executed = False

    # Physical cores in the machine
    num_cores = psutil.cpu_count(logical=False) if config["threads"] == -1 else config["threads"]
    print(f"\nUsing {num_cores} cores\n")

    # Optimisation loop
    while not es.stop() or gen < config["generations"]:
        try:
            # Generate candidate solutions
            X = es.ask()

            # Evaluate in parallel
            if num_cores > 1:
                with Pool(num_cores) as pool:
                    fitvals = pool.map_async(fitness, X).get()
            else:
                fitvals = [fitness(x) for x in X]

            # Correct sign of fevals needed cause this CMA implementation only minimises
            if config["maximise"]:
                fitvals = [-fitval for fitval in fitvals]

            # Inform CMA optimizer of fitness results
            es.tell(X, fitvals)

            if gen % config["print_every"] == 0:
                es.disp()

            # Store best solution
            objective_current_best_sol = es.best.f
            objectives_best.append(objective_current_best_sol)
            if objective_current_best_sol <= objective_solution_best:
                objective_solution_best = objective_current_best_sol
                solution_best = es.best.x

            # Store best mean solution
            objective_current_centroid_sol = np.mean(es.fit.fit)
            objectives_centroid.append(objective_current_centroid_sol)
            if objective_current_centroid_sol <= objective_solution_centroid:
                objective_solution_centroid = objective_current_centroid_sol
                solution_centroid = es.mean

            gen += 1

            if gen % config["evolution_feval_check_every"] == 0:
                test_fevals = []
                print("\n" + 30 * "v")
                for _ in range(config["evolution_feval_check_N"] // config["nb_episode_evals"]):
                    checksum = True if _ == 0 else False
                    # feval = fitness(solution_best, config, render=False, solution_id="best_check_"+str(config['evolution_feval_check_N']), checksum=checksum)
                    feval = fitness(solution_best)
                    test_fevals.append(feval)
                print(f"\nEvaluated {config['nb_episode_evals']*(config['evolution_feval_check_N']//config['nb_episode_evals'])} times the best solution found so far. Mean: {np.mean(test_fevals)}")
                print("\n" + 30 * "^" + "\n")

        # Allows to interrupt optimation with Ctrl+C
        except KeyboardInterrupt:  # Only works with python mp
            time.sleep(5)
            print("\n" + 20 * "*")
            print(f"\nCaught Ctrl+C!\nStopping evolution\n")
            print(20 * "*" + "\n")
            break

        # Early stopping of evolution
        if config["early_stopping"]:
            if gen == config["early_stopping_conditions"]["generation"] and objective_current_best_sol > config["early_stopping_conditions"]["objective_value"]:
                print(f"\nObjective too high {objective_current_best_sol} (reward too low) at generation {gen}.\nUnpromising run! Stopping evolution.\n")
                print(20 * "*")
                early_stopping_executed = True
                break

        # # Stopping evolution if loss flattening
        if config["flattening_stopping"] and gen > config["flattening_stopping_conditions"]["min_generation"]:
            std_objective = np.std(objectives_best[: -config["flattening_stopping_conditions"]["last_generations"]])
            if std_objective < config["flattening_stopping_conditions"]["min_std"]:
                print(f"\nObjective flattening!\nStd last {config['flattening_stopping_conditions']['last_generations']} generation is: {std_objective}\nStopping evolution.\n")
                print(20 * "*")
                break

    # losses/Loss arrays
    objectives_centroid = np.array(objectives_centroid)
    objectives_best = np.array(objectives_best)
    # solution = es.result # unsed since ask&tell

    toc = time.time()
    config["training time"] = str(int(toc - tic)) + " seconds"
    print("\nEvolution took: ", int(toc - tic), " seconds\n")
    print(f"========Optimizer output:==========================")
    print(f"Best single loss found was {objective_solution_best}")
    print(f"Best population centroid loss found was {objective_solution_centroid}")
    print(f"===================================================\n")

    # Create dataframe for logging objective values
    logger_pd = pd.DataFrame({"pop_best_eval": objectives_best, "mean_eval": objectives_centroid})

    return solution_best, solution_centroid, early_stopping_executed, logger_pd




def NEAT(config, fitness):
    """NEAT in place of CMA-ES. Same inputs and outputs as CMAES(config, fitness).
    Each individual = 3 NEAT networks (growth, transform, weight) + initial embedding."""
    import copy
    import random
    from neat_core import RULES, InnovationTracker, new_individual, speciate, remove_stagnant, reproduce

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    pop_size = config["neat_popsize"]
    trackers = {k: InnovationTracker() for k in RULES}
    population = [new_individual(trackers, config["neat_rule_shapes"], config["neat_n_init"])
                  for _ in range(pop_size)]
    species = []
    num_cores = psutil.cpu_count(logical=False) if config["threads"] == -1 else config["threads"]

    # Gates are trained on a softmax score that never reaches exactly 4, so "solved" is checked by counting correct answers
    if "gate" in config["environment"]:
        from train_backend import fitness_functional
        solved_score = fitness_functional({**config, "gate_metric": "accuracy"})
    else:
        solved_score = lambda ind: ind.fitness

    print("\n.......................................................")
    print(f"\nInitialising NEAT with population {pop_size} for {config['neat_generations']} generations")
    print(f"Using {num_cores} cores\n")
    tic = time.time()

    best_ever, objectives_best, objectives_mean = None, [], []
    early_stopping_executed = False
    for gen in range(config["neat_generations"]):
        try:
            # Evaluate (in parallel, like CMAES) with the unchanged NDP fitness function
            if num_cores > 1:
                with Pool(num_cores) as pool:
                    fitvals = pool.map_async(fitness, population).get()
            else:
                fitvals = [fitness(ind) for ind in population]
            for ind, f in zip(population, fitvals):
                ind.fitness = f
                for k in RULES:
                    getattr(ind, k).fitness = f

            gen_best = max(population, key=lambda i: i.fitness)
            if best_ever is None or gen_best.fitness > best_ever.fitness:
                best_ever = copy.deepcopy(gen_best)
            # Log as objectives (negative fitness), the same convention as CMAES
            objectives_best.append(-gen_best.fitness)
            objectives_mean.append(-float(np.mean(fitvals)))

            species = remove_stagnant(speciate(population, species, config["neat_species_threshold"]))
            if gen % config["print_every"] == 0:
                print(f"Gen {gen:4d} | best fitness {gen_best.fitness:5.2f} | best ever {best_ever.fitness:5.2f} | "
                      f"mean {np.mean(fitvals):5.2f} | species {len(species):3d} | t {time.time() - tic:6.1f}s")
            if config.get("neat_stop_when_solved") and solved_score(best_ever) >= config["neat_solved_fitness"]:
                print(f"\nGen {gen} | best individual is solved ({solved_score(best_ever)}). Stopping.")
                break
            population = reproduce(species, pop_size, trackers)

        except KeyboardInterrupt:
            print(f"\nCaught Ctrl+C!\nStopping evolution\n")
            break

    toc = time.time()
    config["training time"] = str(int(toc - tic)) + " seconds"
    print("\nEvolution took: ", int(toc - tic), " seconds\n")
    print(f"========Optimizer output:==========================")
    print(f"Best single loss found was {-best_ever.fitness}")
    print(f"===================================================\n")

    logger_pd = pd.DataFrame({"pop_best_eval": objectives_best, "mean_eval": objectives_mean})
    # NEAT has no population centroid, so the best individual is returned in both slots
    return best_ever, best_ever, early_stopping_executed, logger_pd