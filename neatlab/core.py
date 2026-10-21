"""Small feed-forward NEAT implementation; no third-party dependencies."""
from dataclasses import dataclass, asdict
import copy
import math
import random


@dataclass
class Node:
    id: int
    kind: str


@dataclass
class Gene:
    innovation: int
    source: int
    target: int
    weight: float
    enabled: bool = True


class Innovations:
    """Run-wide homology registry (including repeated splits across generations)."""
    def __init__(self, next_node):
        self.next_node = next_node
        self.edges = {}
        self.splits = {}

    def edge(self, source, target):
        key = (source, target)
        if key not in self.edges:
            self.edges[key] = len(self.edges)
        return self.edges[key]

    def split(self, gene):
        if gene.innovation not in self.splits:
            node = self.next_node
            self.next_node += 1
            self.splits[gene.innovation] = (node, self.edge(gene.source, node), self.edge(node, gene.target))
        return self.splits[gene.innovation]


class Genome:
    def __init__(self, nodes, genes=()):
        self.nodes = {n.id: copy.copy(n) for n in nodes}
        self.genes = {g.innovation: copy.copy(g) for g in genes}
        self.fitness = 0.0

    def clone(self):
        return copy.deepcopy(self)

    def order(self):
        incoming = {n: 0 for n in self.nodes}
        outgoing = {n: [] for n in self.nodes}
        for g in self.genes.values():
            if g.enabled:
                incoming[g.target] += 1
                outgoing[g.source].append(g.target)
        ready = sorted(n for n, degree in incoming.items() if degree == 0)
        result = []
        while ready:
            n = ready.pop(0)
            result.append(n)
            for target in outgoing[n]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
        if len(result) != len(self.nodes):
            raise ValueError('Recurrent connections are not supported')
        return result

    def activate(self, inputs):
        ids = sorted(n.id for n in self.nodes.values() if n.kind == 'input')
        if len(inputs) != len(ids):
            raise ValueError('Incorrect input count')
        values = dict(zip(ids, inputs))
        values.update({n.id: 1.0 for n in self.nodes.values() if n.kind == 'bias'})
        incoming = {n: [] for n in self.nodes}
        for g in self.genes.values():
            if g.enabled:
                incoming[g.target].append(g)
        for n in self.order():
            if n not in values:
                z = sum(values[g.source] * g.weight for g in incoming[n])
                values[n] = 1 / (1 + math.exp(-max(-60, min(60, 4.9*z))))
        return [values[n.id] for n in sorted(self.nodes.values(), key=lambda n:n.id) if n.kind == 'output']

    def can_connect(self, source, target):
        if source == target or self.nodes[target].kind in ('input', 'bias') or self.nodes[source].kind == 'output':
            return False
        stack, seen = [target], set()
        while stack:
            n = stack.pop()
            if n == source:
                return False
            if n not in seen:
                seen.add(n)
                stack.extend(g.target for g in self.genes.values() if g.enabled and g.source == n)
        return True

    def add_node(self, registry, rng):
        candidates = [g for g in self.genes.values() if g.enabled and
                      (g.innovation not in registry.splits or registry.splits[g.innovation][0] not in self.nodes)]
        if not candidates:
            return None
        old = rng.choice(candidates)
        node, first, second = registry.split(old)
        old.enabled = False
        self.nodes[node] = Node(node, 'hidden')
        self.genes[first] = Gene(first, old.source, node, 1.0)
        self.genes[second] = Gene(second, node, old.target, old.weight)
        return {'type': 'add_node', 'split': old.innovation, 'node': node, 'added': [first, second]}

    def add_connection(self, registry, rng):
        existing = {(g.source, g.target):g for g in self.genes.values()}
        pairs = [(a,b) for a in self.nodes for b in self.nodes
                 if ((a,b) not in existing or not existing[a,b].enabled) and self.can_connect(a,b)]
        if not pairs:
            return None
        a,b = rng.choice(pairs)
        innovation = registry.edge(a,b)
        if innovation in self.genes:
            self.genes[innovation].enabled = True
        else:
            self.genes[innovation] = Gene(innovation,a,b,rng.uniform(-2,2))
        return {'type':'add_connection', 'innovation':innovation, 'source':a, 'target':b}

    def data(self):
        return {'nodes':[asdict(n) for n in self.nodes.values()],
                'genes':[asdict(g) for g in self.genes.values()], 'fitness':self.fitness}


def distance(a, b, c1=1.0, c2=1.0, c3=0.4):
    aa, bb = set(a.genes), set(b.genes)
    matching = aa & bb
    boundary = min(max(aa, default=-1), max(bb, default=-1))
    excess = sum(i > boundary for i in aa ^ bb)
    disjoint = len(aa ^ bb) - excess
    n = max(len(aa),len(bb))
    n = n if n >= 20 else 1
    weights = sum(abs(a.genes[i].weight-b.genes[i].weight) for i in matching)/max(1,len(matching))
    return c1*excess/n + c2*disjoint/n + c3*weights


def crossover(a, b, rng):
    """Align historical markings; unmatched genes come from fitter parent.

    On ties both parents' unmatched genes are eligible. A cycle-causing inherited
    connection is disabled, preserving a valid feed-forward phenotype.
    """
    child = Genome(list(a.nodes.values()) + list(b.nodes.values()))
    origins = {}
    for i in sorted(set(a.genes) | set(b.genes)):
        ga, gb = a.genes.get(i), b.genes.get(i)
        if ga and gb:
            origin = rng.choice(['A','B'])
            gene = copy.copy(ga if origin == 'A' else gb)
            if not ga.enabled or not gb.enabled:
                gene.enabled = rng.random() >= 0.75
        else:
            origin = 'A' if ga else 'B'
            if (ga and a.fitness < b.fitness) or (gb and b.fitness < a.fitness):
                continue
            gene = copy.copy(ga or gb)
        if gene.enabled and not child.can_connect(gene.source,gene.target):
            gene.enabled = False
        child.genes[i] = gene
        origins[i] = origin
    used = {g.source for g in child.genes.values()} | {g.target for g in child.genes.values()}
    child.nodes = {i:n for i,n in child.nodes.items() if n.kind != 'hidden' or i in used}
    return child, origins


@dataclass
class Config:
    population: int = 150
    seed: int = 7
    growth: bool = True
    hidden: int = 0
    threshold: float = 3.0
    add_node_rate: float = 0.03
    add_connection_rate: float = 0.08
    mutation_rate: float = 0.8
    crossover_rate: float = 0.75
    stagnation: int = 20

    def __post_init__(self):
        if self.population < 2 or self.hidden < 0 or self.threshold <= 0 or self.stagnation < 1:
            raise ValueError('Invalid population, hidden count, threshold, or stagnation')
        for value in (self.add_node_rate,self.add_connection_rate,self.mutation_rate,self.crossover_rate):
            if not 0 <= value <= 1:
                raise ValueError('Rates must be in [0, 1]')


XOR = [([0,0],0),([0,1],1),([1,0],1),([1,1],0)]


def xor_fitness(genome):
    return (4 - sum((genome.activate(x)[0]-y)**2 for x,y in XOR))**2


class Population:
    def __init__(self, config=None, inputs=2, outputs=1):
        if inputs < 1 or outputs < 1:
            raise ValueError('Need at least one input and output')
        self.config = config or Config()
        self.rng = random.Random(self.config.seed)
        nodes = [Node(i,'input') for i in range(inputs)] + [Node(inputs,'bias')]
        out = list(range(inputs+1,inputs+1+outputs))
        nodes += [Node(i,'output') for i in out]
        hidden = list(range(len(nodes),len(nodes)+self.config.hidden))
        nodes += [Node(i,'hidden') for i in hidden]
        self.registry = Innovations(len(nodes))
        sources = list(range(inputs+1))
        pairs = [(a,b) for a in sources for b in (hidden or out)]
        if hidden:
            pairs += [(a,b) for a in hidden for b in out] + [(inputs,b) for b in out]
        self.genomes = [Genome(nodes,[Gene(self.registry.edge(a,b),a,b,self.rng.uniform(-2,2)) for a,b in pairs])
                        for _ in range(self.config.population)]
        self.species = []
        self.next_species = 0
        self.generation = 0
        self.history = []
        self.best = None
        self.last_mating = None

    def speciate(self):
        for s in self.species:
            s['members'] = []
        for genome in self.genomes:
            candidates = [(distance(genome,s['representative']),s['id'],s) for s in self.species]
            match = min(candidates,default=None,key=lambda x:(x[0],x[1]))
            if match and match[0] < self.config.threshold:
                species = match[2]
            else:
                species = {'id':self.next_species,'representative':genome.clone(), 'members':[], 'best':-math.inf,'stale':0}
                self.next_species += 1
                self.species.append(species)
            species['members'].append(genome)
        self.species = [s for s in self.species if s['members']]
        for s in self.species:
            s['representative'] = self.rng.choice(s['members']).clone()
            best = max(g.fitness for g in s['members'])
            if best > s['best'] + 1e-8:
                s['best'],s['stale'] = best,0
            else:
                s['stale'] += 1

    def evaluate(self, objective=xor_fitness):
        for g in self.genomes:
            g.fitness = float(objective(g))
            if not math.isfinite(g.fitness):
                raise ValueError('Objective must return finite fitness')
        champion = max(self.genomes,key=lambda g:g.fitness)
        if self.best is None or champion.fitness > self.best.fitness:
            self.best = champion.clone()
        self.speciate()
        snapshot = {'generation':self.generation,'best':champion.fitness,
                    'mean':sum(g.fitness for g in self.genomes)/len(self.genomes),
                    'champion':champion.data(), 'record':self.best.fitness,
                    'species':[{'id':s['id'],'size':len(s['members'])} for s in self.species],
                    'mating':self.last_mating}
        self.history.append(snapshot)
        return snapshot

    def mutate(self, genome):
        events = []
        for gene in genome.genes.values():
            if self.rng.random() < self.config.mutation_rate:
                gene.weight = (gene.weight+self.rng.gauss(0,0.5) if self.rng.random()<0.9 else self.rng.uniform(-2,2))
                gene.weight = max(-30,min(30,gene.weight))
        if self.config.growth:
            if self.rng.random()<self.config.add_node_rate:
                events.append(genome.add_node(self.registry,self.rng))
            if self.rng.random()<self.config.add_connection_rate:
                events.append(genome.add_connection(self.registry,self.rng))
        return [e for e in events if e]

    def reproduce(self):
        if not self.species:
            raise ValueError('Evaluate before reproduction')
        champion = max(self.genomes,key=lambda g:g.fitness)
        active = [s for s in self.species if s['stale'] < self.config.stagnation or champion in s['members']]
        # Explicit sharing: sum of fitness / species size gives species mean.
        floor = min(g.fitness for g in self.genomes)
        offset = min(0,floor)
        scores = [sum(g.fitness-offset+1e-6 for g in s['members'])/len(s['members']) for s in active]
        raw = [self.config.population*x/sum(scores) for x in scores]
        quotas = [int(x) for x in raw]
        for i in sorted(range(len(raw)),key=lambda i:raw[i]-quotas[i],reverse=True)[:self.config.population-sum(quotas)]:
            quotas[i] += 1
        champ_index = next(i for i,s in enumerate(active) if champion in s['members'])
        if quotas[champ_index] == 0:
            donor = max(range(len(quotas)),key=lambda i:quotas[i])
            quotas[donor] -= 1
            quotas[champ_index] = 1
        children, matings = [], []
        for s,quota in zip(active,quotas):
            if not quota:
                continue
            ranked = sorted(s['members'],key=lambda g:g.fitness,reverse=True)
            children.append(ranked[0].clone())
            pool = ranked[:max(2,math.ceil(len(ranked)*0.5))]
            for _ in range(quota-1):
                a = self.rng.choice(pool)
                if len(pool)>1 and self.rng.random()<self.config.crossover_rate:
                    b = self.rng.choice([p for p in pool if p is not a])
                    child,origins = crossover(a,b,self.rng)
                    before = child.data()
                    events = self.mutate(child)
                    matings.append({'a':a.data(),'b':b.data(),'child':before,'mutated':child.data(),'origins':origins,'events':events})
                else:
                    child = a.clone()
                    self.mutate(child)
                child.fitness = 0.0
                children.append(child)
        self.last_mating = next((m for m in matings if m['events']), matings[0] if matings else None)
        self.genomes = children
        self.species = active
        self.generation += 1
