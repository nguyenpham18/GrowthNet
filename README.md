# GrowthNet

## NEAT evolution laboratory

A dependency-free Python 3.10+ implementation of a **feed-forward NEAT baseline**, with a portable interactive HTML replay. Standard topology evolution is enabled; biological/developmental growth rules are deliberately deferred.

## Run

From this directory:

```sh
python3 -m neatlab --seed 7 --population 150 --generations 150
python3 -m unittest discover -s tests -v
```

Open `output/report.html` in a browser. It contains all its data and works offline; no server or installation is needed. Drag the generation slider, play the replay, or jump to the final generation. The report shows the champion network, XOR predictions, fitness curves, species history, and an actual sampled parent/offspring crossover for each generation. Hover over edges for weights and innovation IDs. The mating table distinguishes inherited genes from subsequent mutation; the sampled child need not be the generation champion.

Outputs (overwritten on each run unless `--output` is changed):

- `output/report.html`: interactive run replay.
- `output/run.json`: configuration, champion history, species sizes, sampled mating, and best-ever genome.
- `output/champion.json`: best-ever genome, with node and connection genes.

For a fixed-topology control experiment:

```sh
python3 -m neatlab --fixed-topology --hidden 2 --output output/fixed
```

A minimal fixed network with no hidden nodes cannot solve XOR. By default, growth mode starts with two inputs, a bias, one output, and three connections. Generations count evaluated populations, including generation zero. Runs continue for the requested count even after solving XOR. Fitness is `(4 - sum_squared_error)^2`, maximum 16; “solved” requires every absolute output error to be below 0.1. Evolution is stochastic; different seeds can fail within a finite budget.

## Core mechanisms

1. **Genome:** explicit node genes and connection genes with weight, enabled flag, and innovation ID. A constant bias node supplies trainable offsets through outgoing connections. Hidden/output activations are `sigmoid(4.9 * weighted_sum)`.
2. **Historical markings:** one registry aligns homologous connections and node splits throughout the run. Splitting disables the old edge and inserts a hidden node with incoming weight 1 and outgoing weight equal to the old edge. This does not exactly preserve the network function because a sigmoid is inserted.
3. **Structural mutation:** split enabled edges and add/re-enable cycle-safe edges. Inputs/bias cannot receive connections; outputs cannot send them. Weight mutation perturbs or replaces weights, with a magnitude cap of 30.
4. **Crossover:** matching genes are sampled from either parent; unmatched genes come from the fitter parent. On ties, unmatched genes from both parents are included. If either matching gene is disabled, the child gene is disabled with probability 0.75. Cycle-producing inherited edges are retained as disabled genes. Parent objects are never modified.
5. **Speciation:** compatibility distance uses excess, disjoint, and mean matching-weight differences (including disabled genes). Coefficients are 1, 1, 0.4; genome-size normalization is 1 below 20 genes. Representatives persist between generations, with nearest compatible assignment and a configurable fixed threshold.
6. **Selection:** explicit sharing gives each species a reproductive score equal to its mean fitness. Scores are shifted when an objective is negative. Largest-remainder allocation preserves exact population size. Each allocated species keeps its champion, parents are sampled from its top half, and stagnant species expire after 20 generations without improvement (the global champion's species is protected).

## Extend the objective

```python
from neatlab import Config, Population

population = Population(Config(seed=11), inputs=3, outputs=2)

def objective(genome):
    outputs = genome.activate([0.2, 0.4, 0.8])
    return -sum(value**2 for value in outputs)  # finite scalar; larger is better

for generation in range(100):
    population.evaluate(objective)
    if generation < 99:
        population.reproduce()

print(population.best.data())
```

Call `evaluate` exactly once per generation before `reproduce`. The HTML renderer and CLI are specifically for the 2-input/1-output XOR experiment; the Python core supports other input/output dimensions and objectives. Saved JSON is an analysis export, not a resumable evolutionary checkpoint: it does not preserve the full population, RNG, or species state.

## Scope and references

This is a compact educational variant of NEAT, not a bit-for-bit reproduction of the 2002 experiments. Simplifications include feed-forward graphs only, run-wide innovation reuse (rather than generation-local duplicate-event tracking), fixed species threshold, top-half uniform parent selection, no interspecies mating, and one saved mating example per generation. It is intended for small research experiments, not optimized for large networks.

- Stanley & Miikkulainen (2002), [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf): historical markings, structural evolution, and speciation.
- Najarro, Sudhakaran & Risi (2023), *Towards Self-Assembling Artificial Neural Networks through Neural Developmental Programs*, supplied as `isal_a_00697.pdf`: related research direction. In that paper, a shared program grows a network within its lifetime. This baseline directly evolves network genomes between generations; it does not reproduce that paper's developmental algorithm or experimental results.

A future developmental experiment can evaluate a genome as a controller for local growth rules inside the objective. The current implementation deliberately keeps that developmental process separate from evolutionary mutation.

## Validation performed

Ten unit tests pass, covering split homology, disabled inheritance, fitter-parent crossover, compatibility distance, cycle protection, population size, fixed topology, deterministic runs, invalid objectives, and a hand-constructed XOR network. Browser replay, final-generation navigation, gene tables, and before/after mutation rendering were checked; the browser reported no JavaScript errors.

At 150 individuals and 150 evaluated generations, seeds 1, 3, and 7 solved XOR. Seed 2 reached fitness 15.8213 but missed the strict error threshold (maximum error 0.1376). The bundled seed-7 example reached fitness 15.999997 with maximum error 0.000568. A fixed-topology control with two hidden nodes and seed 7 solved XOR in a 100-generation run. These small smoke experiments are not a statistical performance benchmark. Additional-seed results are saved in `output/validation.json`.
