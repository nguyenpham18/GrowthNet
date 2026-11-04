"""Tie check for a saved NDP XOR solution (CMA-ES or NEAT).
Usage (from the ndp folder):  python check_ties.py saved_models/<Model ID>
Re-grows the saved solution with THEIR code and prints both output values for each XOR input.
Honest scoring: an answer counts only if the right output is bigger by more than 1e-6.
"""
import sys, re, copy, yaml
import numpy as np, networkx as nx, torch
torch.set_default_dtype(torch.float64)
sys.path.insert(0, '.')
import train_backend
from NDP import MLP, propagate_features, predict_new_nodes, add_new_nodes, update_weights
from train_backend import bool_gates_fitness, train_model

run_dir = sys.argv[1].rstrip('/')
saved_cfg = open(run_dir + '/config.yml').read()
config = yaml.safe_load(open('experiments_paper/XOR/run_experiment.yaml'))
m = re.search(r'^seed:\s*(\d+)', saved_cfg, re.M)
config['seed'] = int(m.group(1)) if m else 0                    # same seed -> same initial graph
EMBRYO = bool(re.search(r'^initial_graph:\s*embryo', saved_cfg, re.M))
if EMBRYO:
    config['initial_graph'] = 'embryo'
train_backend.CMAES = lambda config, fitness: (None, None, False, None)   # their setup, no training
train_model(config)
EMB = config['node_embedding_size']

# ---- Load the saved solution: CMA-ES = 14 numbers, NEAT = a saved Individual
sol = np.load(run_dir + '/solution_best.npy', allow_pickle=True)
if sol.dtype == object:                                          # NEAT
    from neat_core import rule_nets
    ind = sol.item()
    growth, transform, weight = rule_nets(ind, config)
    init_state = np.array(ind.init_emb, dtype=float)
    print("\nOptimizer: NEAT" + (" | start: embryo" if EMBRYO else " | start: 1 seed node"))
else:                                                            # CMA-ES
    vec = sol
    models = [MLP(EMB, 1, config['mlp_growth_hidden_layers_dims'], torch.nn.Tanh(),
                  config['growth_model_last_layer_activated'], config['growth_model_bias']),
              MLP(EMB, EMB, config['mlp_embedding_transform_hidden_layers_dims'], torch.nn.Tanh(),
                  config['transform_model_last_layer_activated'], config['transform_model_bias']),
              MLP(2 * EMB, 1, config['mlp_weight_values_hidden_layers_dims'], torch.nn.Tanh(),
                  config['mlp_weight_values_last_layer_activated'], config['mlp_weight_values_bias'])]
    i = config['nb_params_coevolve_initial_embeddings']
    for mdl, n in zip(models, [config['nb_params_growth_model'], config['nb_params_feature_transformation'],
                               config['nb_params_mlp_weight_values']]):
        torch.nn.utils.vector_to_parameters(torch.tensor(vec[i:i + n], dtype=torch.float64), mdl.parameters())
        i += n
    growth, transform, weight = models
    init_state = np.array(vec[:config['nb_params_coevolve_initial_embeddings']], dtype=float)
    print("\nOptimizer: CMA-ES" + (" | start: embryo" if EMBRYO else " | start: 1 seed node"))

# ---- Their growth loop (copied from train_backend.fitness_functional, drawing code removed)
G = copy.deepcopy(config['shared_intial_graph'])
state = init_state.reshape(-1, EMB)
for _ in range(config['number_of_growth_cycles']):
    d = nx.diameter(G.to_undirected())
    state = propagate_features(state, nx.to_numpy_array(G), d + config['network_thinking_time_extra_growth'],
                               config['recurrent_activation_function'], config['additive_update'],
                               None, transform)
    new = predict_new_nodes(growth, state, EMB)
    if EMBRYO:                                                   # only hidden nodes divide
        new = np.array(new, dtype=bool).reshape(-1)
        new[config['input_nodes'] + config['output_nodes']] = False
    G, state = add_new_nodes(copy.deepcopy(G), config, state, None, new, True, False)
    if len(G) > 1:
        G = update_weights(G, state, weight, config)

if len(G) < config['min_network_size']:
    print(f"{len(G)} nodes: too small, their fitness = {len(G) - config['min_network_size']}")
    sys.exit()
print(f"{len(G)} nodes, {G.number_of_edges()} edges | their fitness = {bool_gates_fitness(G, config)}\n")

# ---- Re-run their XOR evaluation loop, printing both outputs (state is not reset between inputs, like theirs)
W = nx.to_numpy_array(G)
steps = nx.diameter(G.to_undirected()) + config['network_thinking_time_extra_rollout']
s = np.zeros(len(G)); honest = 0
for x, y in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1, 1, 0]):
    s[:2] = x
    s = propagate_features(s, W, steps, config['recurrent_activation_function'],
                           config['additive_update'], np.array(x, dtype=float))
    o0, o1 = (s[config['output_nodes']] if EMBRYO else s[-2:])
    ok = (o1 - o0 if y == 1 else o0 - o1) > 1e-6
    honest += ok
    print(f"x={x} want {y} | out0={o0:+.17f} out1={o1:+.17f} | diff={o1 - o0:+.2e} | "
          f"argmax={int(np.argmax([o0, o1]))} | honest {'OK' if ok else 'WRONG'}")
print(f"\nhonest score (ties count as wrong) = {honest}/4")
out_ids = config['output_nodes'] if EMBRYO else [len(G) - 2, len(G) - 1]
print("output-node embeddings:", state[out_ids[0]], state[out_ids[1]])