import random
import unittest
from neatlab.core import *


class CoreTests(unittest.TestCase):
    def test_split_homology_and_no_aliasing(self):
        p = Population(Config(population=2))
        a,b = p.genomes
        # One enabled edge ensures the same split in both genomes.
        for g in (a,b):
            for edge in g.genes.values():
                edge.enabled = edge.innovation == 0
        original = a.genes[0].weight
        ea = a.add_node(p.registry,random.Random(1))
        eb = b.add_node(p.registry,random.Random(1))
        self.assertEqual(ea,eb)
        self.assertFalse(a.genes[0].enabled)
        self.assertEqual(a.genes[ea['added'][0]].weight,1)
        self.assertEqual(a.genes[ea['added'][1]].weight,original)
        a.genes[0].weight = 999
        self.assertNotEqual(b.genes[0].weight,999)
        a.order()

    def test_distance_excess_disjoint_and_weights(self):
        nodes = [Node(0,'input'),Node(1,'output')]
        a = Genome(nodes,[Gene(0,0,1,1),Gene(2,0,1,2)])
        b = Genome(nodes,[Gene(0,0,1,2),Gene(1,0,1,2),Gene(3,0,1,2)])
        self.assertAlmostEqual(distance(a,b),3.4)
        self.assertEqual(distance(a,a),0)
        self.assertEqual(distance(a,b),distance(b,a))

    def test_fitter_parent_and_gene_alignment(self):
        p = Population(Config(population=2))
        a,b = p.genomes
        a.add_node(p.registry,random.Random(3))
        a.fitness,b.fitness = 2,1
        child,origins = crossover(a,b,random.Random(9))
        self.assertEqual(set(child.genes),set(a.genes))
        for i in set(a.genes)-set(b.genes):
            self.assertEqual(origins[i],'A')
        for i in set(a.genes)&set(b.genes):
            parent = a if origins[i]=='A' else b
            self.assertEqual(child.genes[i].weight,parent.genes[i].weight)
        child.genes[0].weight = 100
        self.assertNotEqual(a.genes[0].weight,100)

    def test_disabled_inheritance_probability(self):
        p = Population(Config(population=2))
        a,b = p.genomes
        a.genes[0].enabled = False
        rng = random.Random(12)
        disabled = sum(not crossover(a,b,rng)[0].genes[0].enabled for _ in range(1000))
        self.assertTrue(680 < disabled < 820)

    def test_crossover_cycle_guard(self):
        nodes=[Node(0,'input'),Node(1,'hidden'),Node(2,'hidden'),Node(3,'output')]
        a=Genome(nodes,[Gene(0,1,2,1)])
        b=Genome(nodes,[Gene(1,2,1,1)])
        child,_=crossover(a,b,random.Random(1))
        child.order()
        self.assertEqual(sum(g.enabled for g in child.genes.values()),1)

    def test_fixed_topology_and_population_size(self):
        p=Population(Config(population=31,growth=False,hidden=2))
        ids=set(p.genomes[0].genes)
        for _ in range(12):
            p.evaluate()
            p.reproduce()
            self.assertEqual(len(p.genomes),31)
            for g in p.genomes:
                self.assertEqual(set(g.genes),ids)
                self.assertEqual(len(g.nodes),6)

    def test_growth_graph_invariants(self):
        p=Population(Config(population=25,add_node_rate=.5,add_connection_rate=.8))
        for _ in range(12):
            p.evaluate()
            p.reproduce()
            self.assertEqual(len(p.genomes),25)
            for g in p.genomes:
                g.order()
                self.assertEqual(len({(e.source,e.target) for e in g.genes.values()}),len(g.genes))
                self.assertTrue(0 <= g.activate([0,1])[0] <= 1)

    def test_reproducibility_and_negative_objectives(self):
        histories=[]
        for _ in range(2):
            p=Population(Config(population=20,seed=42))
            for _ in range(5):
                p.evaluate(lambda g:-xor_fitness(g))
                p.reproduce()
            histories.append(p.history)
        self.assertEqual(histories[0],histories[1])

    def test_invalid_config_and_inputs(self):
        with self.assertRaises(ValueError): Config(population=1)
        with self.assertRaises(ValueError): Config(add_node_rate=2)
        p=Population(Config(population=2))
        with self.assertRaises(ValueError): p.genomes[0].activate([1])
        with self.assertRaises(ValueError): p.evaluate(lambda g:float('nan'))

    def test_known_xor_network(self):
        # Hidden units approximate OR and NAND; output approximates AND.
        nodes=[Node(0,'input'),Node(1,'input'),Node(2,'bias'),Node(3,'output'),Node(4,'hidden'),Node(5,'hidden')]
        edges=[(0,4,10),(1,4,10),(2,4,-5),(0,5,-10),(1,5,-10),(2,5,15),(4,3,10),(5,3,10),(2,3,-15)]
        g=Genome(nodes,[Gene(i,*edge) for i,edge in enumerate(edges)])
        self.assertGreater(xor_fitness(g),15.99)


if __name__ == '__main__':
    unittest.main()
