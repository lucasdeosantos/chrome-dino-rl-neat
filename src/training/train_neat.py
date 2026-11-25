import neat
import pickle
from copy import deepcopy
from shutil import copyfile
from ..utils import write_json
from .checkpoints import read_checkpoint, save_checkpoint, restore_records
from .agents import act, neat_network, save_neat
from .env import DinoEnv
from .evaluation import evaluate
from .renderer import TrainingRenderer, StopRendering


def load_neat_config(path):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, str(path))
    return config


def load_start_config(args):
    if args.pretrained is None:
        return load_neat_config(args.neat_config)
    
    with args.pretrained.open('rb') as file:
        config = pickle.load(file)['config']
    
    return config


def create_population(args):
    config = load_start_config(args)
    
    if args.population is not None:
        config.pop_size = args.population

    tracker = getattr(config.genome_config, 'innovation_tracker', None)
    population = neat.Population(config, seed=args.seed)
    
    if args.pretrained is not None:
        with args.pretrained.open('rb') as file:
            source = pickle.load(file)['genome']
        
        if tracker is not None:
            population.reproduction.innovation_tracker = tracker
            config.genome_config.innovation_tracker = tracker
        
        for index, key in enumerate(population.population):
            genome = deepcopy(source)
            genome.key = key
            genome.fitness = None
            if index:
                genome.mutate(config.genome_config)
            population.population[key] = genome
        
        population.species = config.species_set_type(config.species_set_config, population.reporters)
        population.species.speciate(config, population.population, 0)
    
    return population


def evaluate_population(genomes, config, scenario, max_steps, seed, viewer):
    envs, networks, observations = {}, {}, {}
    results = {}
    partial_individuals = []
    
    try:
        for key, genome in genomes:
            envs[key] = DinoEnv(scenario, max_steps=max_steps)
            observations[key], _ = envs[key].reset(seed=seed)
            networks[key] = neat_network(genome, config)
        
        active = list(envs)
        while active and viewer.poll():
            finished = []
            state = None
            for key in active:
                env = envs[key]
                action = act('neat', networks[key], observations[key])
                observations[key], _, terminated, truncated, info = env.step(action)
                state = env.game.get_state()
                if terminated or truncated:
                    results[key] = {'seed': seed, 'steps': env._elapsed_steps, **info}
                    finished.append(key)
            active = [key for key in active if key not in finished]
            live_states = [envs[key].game.get_state() for key in active]
            viewer.step(live_states[0] if live_states else state, population=live_states, force=not active)
        
        if viewer.stopped:
            partial_individuals = [{'individual': key, 'seed': seed, 'steps': envs[key]._elapsed_steps,
                                       'points': envs[key].game.get_points()} for key in active]
        
        return results, partial_individuals
    finally:
        for env in envs.values():
            env.close()


def train_neat(args):
    print("Starting training with NEAT...", flush=True)
    print(f'Scenario: {args.scenario} | max steps: {args.max_steps}', flush=True)
    state = read_checkpoint(args.output) if args.resume is not None else None
    
    if state is None:
        population = create_population(args)
        population.add_reporter(neat.Checkpointer(generation_interval=1, time_interval_seconds=None,
            filename_prefix=str(args.output / 'neat-checkpoint-')))
    else:
        population = state['population']
        restore_records(args, state)
        print(f'Resuming at generation {population.generation + 1}', flush=True)

    for reporter in population.reporters.reporters:
        if isinstance(reporter, neat.Checkpointer):
            reporter.filename_prefix = str(args.output / 'neat-checkpoint-')
    
    config = population.config
    config.save(str(args.output / 'effective_neat_config.txt'))
    viewer = TrainingRenderer(args.render) if args.render else None
    progress = state or {}
    history = progress.get('history', [])
    best = progress.get('best')
    best_score = progress.get('best_score', float('-inf'))
    best_reached_limit = progress.get('best_reached_limit', False)
    success = progress.get('success', False)
    completed_generations = progress.get('completed_generations', 0)
    total_steps = progress.get('total_steps', 0)
    partial_individuals = []

    def checkpoint():
        save_checkpoint(args, dict(population=population, history=history, best=best,
                        best_score=best_score, best_reached_limit=best_reached_limit,
                        evaluated=len(history), success=success,
                        completed_generations=completed_generations, total_steps=total_steps))

    def record_result(genome, result):
        nonlocal success, total_steps
        total_steps += result['steps']
        genome.fitness = result['points']
        success = success or result['reached_limit']
        record = {'generation': population.generation + 1, 'individual': genome.key, **result}
        history.append(record)
        return record

    def evaluate_genomes(genomes, config):
        nonlocal completed_generations, partial_individuals, best, best_score, best_reached_limit
        map_seed = args.training_seed_start + population.generation
        records = []
        
        if viewer is None:
            for _, genome in genomes:
                result = evaluate('neat', neat_network(genome, config), args.scenario, args.max_steps, [map_seed], include_steps=True)['results'][0]
                records.append(record_result(genome, result))
        else:
            results, partial_individuals = evaluate_population(genomes, config, args.scenario, args.max_steps, map_seed, viewer)
            for key, genome in genomes:
                if key in results:
                    records.append(record_result(genome, results[key]))
            if len(results) != len(genomes):
                write_json(args.output / 'training_episodes.json', history)
                raise StopRendering()

        index = max(range(len(records)), key=lambda i: (records[i]['reached_limit'], records[i]['points'], i))
        winner = genomes[index][1]
        record = records[index]
        save_neat(winner, config, args.output / 'last.pkl')
        
        if (record['reached_limit'], record['points']) >= (best_reached_limit, best_score):
            best = deepcopy(winner)
            best_score = record['points']
            best_reached_limit = record['reached_limit']
            copyfile(args.output / 'last.pkl', args.output / 'best.pkl')
            write_json(args.output / 'best_training.json', record)
        
        completed_generations += 1
        write_json(args.output / 'training_episodes.json', history)
        print(f'generation: {population.generation + 1} | seed: {map_seed}'
              f" | best score: {record['points']}", flush=True)
        
        if viewer is not None and viewer.stopped:
            raise StopRendering()

    try:
        try:
            if state is None:
                checkpoint()
            
            while population.generation < args.generations:
                if args.stop_at_max_steps and success:
                    break
                
                population.run(evaluate_genomes, 1)
                checkpoint()
        
        except StopRendering:
            pass
        
        interrupted = viewer is not None and viewer.stopped
        if interrupted:
            total_steps += sum(item['steps'] for item in partial_individuals)
        
        print(f'Completed generations: {completed_generations}/{args.generations}')
        print(f'Interrupted: {interrupted}')
        print(f'Total training steps: {total_steps}', flush=True)
        
        stop_reason = 'interrupted' if interrupted else 'generation_limit'
        if args.stop_at_max_steps and success and not interrupted:
            stop_reason = 'max_steps_reached'
       
        write_json(args.output / 'training_episodes.json', history)
        write_json(args.output / 'training_result.json',
                   {'agent': 'neat', 'completed_steps': total_steps, 'completed_generations': completed_generations,
                    'generation_reached': population.generation + (1 if interrupted else 0),
                    'evaluated_individuals': len(history), 'best_points': best_score if best is not None else None,
                    'success': success, 'partial_individuals': partial_individuals if interrupted else None,
                    'stop_reason': stop_reason})
        return best
    finally:
        if viewer is not None:
            viewer.close()


