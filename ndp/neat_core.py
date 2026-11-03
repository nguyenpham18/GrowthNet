"""NEAT for NDP: evolves the 3 NDP rules (growth, transform, weight) as NEAT networks.

The only thing this replaces is the optimizer (CMA-ES). A NEAT network stands in for each of
NDP's 3 small MLPs, built to behave like them: tanh hidden units, a bias input, and the
output activation taken from the NDP config.
"""
import copy
import random
import numpy as np
import torch

RULES = ("growth", "transform", "weight")


# ---------------------------------------------------------------- NEAT genome
class InnovationTracker:
    """Same (in, out) connection always gets the same innovation number."""
    def __init__(self):
        self.next_innovation, self.history, self.split_history = 0, {}, {}

    def get(self, a, b):
        if (a, b) not in self.history:
            self.history[(a, b)] = self.next_innovation
            self.next_innovation += 1
        return self.history[(a, b)]


class NodeGene:
    def __init__(self, nid, kind):
        self.id, self.type = nid, kind          # kind: 'input', 'output', 'hidden'


class ConnectionGene:
    def __init__(self, a, b, w, enabled, innov):
        self.in_node, self.out_node, self.weight, self.enabled, self.innovation = a, b, w, enabled, innov


class Genome:
    def __init__(self):
        self.nodes, self.connections, self.fitness = {}, {}, 0.0


def minimal_genome(n_in, n_out, tracker):
    """Start minimal: every input wired to every output, no hidden nodes."""
    g = Genome()
    for i in range(n_in):
        g.nodes[i] = NodeGene(i, "input")
    for o in range(n_in, n_in + n_out):
        g.nodes[o] = NodeGene(o, "output")
    for i in range(n_in):
        for o in range(n_in, n_in + n_out):
            k = tracker.get(i, o)
            g.connections[k] = ConnectionGene(i, o, random.uniform(-1, 1), True, k)
    return g


# ---------------------------------------------------------------- mutation / crossover
def mutate_weights(g, perturb=0.9, step=0.5):
    for c in g.connections.values():
        c.weight = c.weight + random.gauss(0, step) if random.random() < perturb else random.uniform(-2, 2)


def creates_cycle(g, a, b):
    stack, seen = [b], set()
    while stack:
        n = stack.pop()
        if n == a:
            return True
        if n in seen:
            continue
        seen.add(n)
        stack += [c.out_node for c in g.connections.values() if c.in_node == n]
    return False


def mutate_add_connection(g, tracker, tries=20):
    nodes = list(g.nodes.values())
    existing = {(c.in_node, c.out_node) for c in g.connections.values()}
    for _ in range(tries):
        a, b = random.choice(nodes), random.choice(nodes)
        if a.type == "output" or b.type == "input" or a.id == b.id:
            continue
        if (a.id, b.id) in existing or creates_cycle(g, a.id, b.id):
            continue
        k = tracker.get(a.id, b.id)
        g.connections[k] = ConnectionGene(a.id, b.id, random.uniform(-1, 1), True, k)
        return


def mutate_add_node(g, tracker):
    on = [c for c in g.connections.values() if c.enabled]
    if not on:
        return
    old = random.choice(on)
    old.enabled = False
    if old.innovation not in tracker.split_history:
        tracker.split_history[old.innovation] = max(set(g.nodes) | set(tracker.split_history.values())) + 1
    nid = tracker.split_history[old.innovation]
    g.nodes[nid] = NodeGene(nid, "hidden")
    k1, k2 = tracker.get(old.in_node, nid), tracker.get(nid, old.out_node)
    g.connections[k1] = ConnectionGene(old.in_node, nid, 1.0, True, k1)
    g.connections[k2] = ConnectionGene(nid, old.out_node, old.weight, True, k2)


def crossover(p1, p2):
    """Child gets the fitter parent's structure; matching genes are picked at random."""
    if p2.fitness > p1.fitness:
        p1, p2 = p2, p1
    child = Genome()
    for k, c1 in p1.connections.items():
        if k in p2.connections:
            c2 = p2.connections[k]
            ch = random.choice([c1, c2])
            en = True if (c1.enabled and c2.enabled) else random.random() > 0.75
        else:
            ch, en = c1, c1.enabled
        child.connections[k] = ConnectionGene(ch.in_node, ch.out_node, ch.weight, en, k)
    for nid, n in p1.nodes.items():
        child.nodes[nid] = NodeGene(nid, n.type)
    return child


def distance(g1, g2, c1=1.0, c2=1.0, c3=0.4):
    k1, k2 = set(g1.connections), set(g2.connections)
    match, other = k1 & k2, (k1 | k2) - (k1 & k2)
    E = sum(1 for k in other if k > min(max(k1), max(k2)))
    D = len(other) - E
    W = sum(abs(g1.connections[k].weight - g2.connections[k].weight) for k in match) / len(match) if match else 0
    N = max(len(k1), len(k2))
    N = 1 if N < 20 else N
    return c1 * E / N + c2 * D / N + c3 * W


# ---------------------------------------------------------------- NEAT network acting like an NDP MLP
class RuleNet:
    """Runs a NEAT genome like one of NDP's MLPs: takes and returns torch tensors,
    works on one row or many rows, tanh hidden units, last input is a bias of 1."""
    def __init__(self, genome, out_activated):
        self.ins = sorted(n.id for n in genome.nodes.values() if n.type == "input")
        self.outs = sorted(n.id for n in genome.nodes.values() if n.type == "output")
        self.out_activated = out_activated
        inc = {n: [] for n in genome.nodes}
        for c in genome.connections.values():
            if c.enabled:
                inc[c.out_node].append((c.in_node, c.weight))
        done, todo, self.order = set(self.ins), [n for n in genome.nodes if n not in self.ins], []
        while todo:
            ready = [n for n in todo if all(a in done for a, _ in inc[n])]
            for n in ready:
                self.order.append((n, inc[n]))
                done.add(n)
                todo.remove(n)

    def __call__(self, x):
        arr = x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)
        single = arr.ndim == 1
        X = np.atleast_2d(arr).astype(float)
        X = np.hstack([X, np.ones((len(X), 1))])          # bias input
        v = {nid: X[:, k] for k, nid in enumerate(self.ins)}
        for nid, inc in self.order:
            s = sum((v[a] * w for a, w in inc), np.zeros(len(X)))
            v[nid] = np.tanh(s) if (nid not in self.outs or self.out_activated) else s
        out = np.stack([v[o] for o in self.outs], axis=1)
        return torch.tensor(out[0] if single else out, dtype=torch.float64)


# ---------------------------------------------------------------- one individual = 3 rules + initial embedding
class Individual:
    def __init__(self, growth, transform, weight, init_emb):
        self.growth, self.transform, self.weight = growth, transform, weight
        self.init_emb = list(init_emb)
        self.fitness = 0.0


def new_individual(trackers, emb, n_init=None):
    """n_init = how many starting-embedding numbers (emb for 1 seed node, emb x 5 for the embryo)."""
    return Individual(minimal_genome(emb + 1, 1, trackers["growth"]),        # +1 = bias input
                      minimal_genome(emb + 1, emb, trackers["transform"]),
                      minimal_genome(2 * emb + 1, 1, trackers["weight"]),
                      [random.uniform(-1, 1) for _ in range(n_init or emb)])  # like x0 ~ U[-1, 1]        # like x0 ~ U[-1, 1]


def rule_nets(ind, config):
    """The 3 NEAT rules, in the same order/roles as NDP's 3 MLPs."""
    return (RuleNet(ind.growth, config["growth_model_last_layer_activated"]),
            RuleNet(ind.transform, config["transform_model_last_layer_activated"]),
            RuleNet(ind.weight, config["mlp_weight_values_last_layer_activated"]))


def mutate_ind(ind, trackers):
    for k in RULES:
        g = getattr(ind, k)
        if random.random() < 0.8:
            mutate_weights(g)
        if random.random() < 0.05:
            mutate_add_node(g, trackers[k])
        if random.random() < 0.1:
            mutate_add_connection(g, trackers[k])
    if random.random() < 0.8:
        ind.init_emb = [v + random.gauss(0, 0.5) for v in ind.init_emb]


def mate(p1, p2):
    return Individual(*(crossover(getattr(p1, k), getattr(p2, k)) for k in RULES),
                      init_emb=random.choice([p1, p2]).init_emb)


# ---------------------------------------------------------------- species
class Species:
    def __init__(self, sid, rep):
        self.id, self.representative, self.members = sid, rep, []


def speciate(pop, species, threshold, next_id=[0]):
    for s in species:
        s.members = []
    for ind in pop:
        for s in species:
            if sum(distance(getattr(ind, k), getattr(s.representative, k)) for k in RULES) < threshold:
                s.members.append(ind)
                break
        else:
            s = Species(next_id[0], ind)
            next_id[0] += 1
            s.members.append(ind)
            species.append(s)
    species[:] = [s for s in species if s.members]
    for s in species:
        s.representative = random.choice(s.members)
    return species


def remove_stagnant(species, limit=15):
    for s in species:
        b = max(i.fitness for i in s.members)
        if b > getattr(s, "best", -np.inf):
            s.best, s.stagnant = b, 0
        else:
            s.stagnant = getattr(s, "stagnant", 0) + 1
    ranked = sorted(species, key=lambda s: s.best, reverse=True)
    return [s for s in ranked if s in ranked[:2] or s.stagnant < limit]


def reproduce(species, pop_size, trackers, survival=0.2):
    low = min(i.fitness for s in species for i in s.members)
    for s in species:                                   # fitness sharing (shifted to be > 0)
        for i in s.members:
            i.adjusted = (i.fitness - low + 1) / len(s.members)
    totals = [sum(i.adjusted for i in s.members) for s in species]
    counts = [int(pop_size * t / sum(totals)) for t in totals]
    counts[totals.index(max(totals))] += pop_size - sum(counts)
    new = []
    for s, n in zip(species, counts):
        if n == 0:
            continue
        ranked = sorted(s.members, key=lambda i: i.fitness, reverse=True)
        if len(ranked) > 5:
            new.append(copy.deepcopy(ranked[0]))        # elitism
            n -= 1
        parents = ranked[:max(1, int(len(ranked) * survival))]
        for _ in range(n):
            if random.random() < 0.25 or len(parents) == 1:
                child = copy.deepcopy(random.choice(parents))
            else:
                child = mate(*random.sample(parents, 2))
            mutate_ind(child, trackers)
            new.append(child)
    return new