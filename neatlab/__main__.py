"""Run an XOR experiment and write portable HTML/JSON artifacts."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .core import Config, Population, XOR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generations',type=int,default=150)
    parser.add_argument('--population',type=int,default=150)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--fixed-topology',action='store_true')
    parser.add_argument('--hidden',type=int,default=0,help='Initial hidden nodes (use 2+ for fixed-topology XOR)')
    parser.add_argument('--output',type=Path,default=Path('output'))
    args = parser.parse_args()
    if args.generations < 1:
        parser.error('--generations must be positive')
    try:
        config = Config(population=args.population,seed=args.seed,growth=not args.fixed_topology,hidden=args.hidden)
    except ValueError as exc:
        parser.error(str(exc))
    pop = Population(config)
    for generation in range(args.generations):
        snapshot = pop.evaluate()
        if generation % 25 == 0 or generation == args.generations-1:
            print(f"Generation {generation:3}: best={snapshot['best']:.5f}, species={len(pop.species)}")
        if generation < args.generations-1:
            pop.reproduce()
    solved = all(abs(pop.best.activate(x)[0]-y)<0.1 for x,y in XOR)
    data = {'config':asdict(config),'history':pop.history,'best':pop.best.data(),'solved':solved}
    args.output.mkdir(parents=True,exist_ok=True)
    payload = json.dumps(data,separators=(',',':'),allow_nan=False)
    (args.output/'run.json').write_text(payload)
    template = Path(__file__).with_name('report.html').read_text()
    (args.output/'report.html').write_text(template.replace('__DATA__',payload))
    (args.output/'champion.json').write_text(json.dumps(pop.best.data(),indent=2))
    print(f"Best ever: {pop.best.fitness:.6f}; solved: {solved}")
    for x,y in XOR:
        print(f'{x} -> {pop.best.activate(x)[0]:.6f} (target {y})')
    print(f'Report: {(args.output/"report.html").resolve()}')


if __name__ == '__main__':
    main()
