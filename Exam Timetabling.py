import numpy as np
import random
import matplotlib.pyplot as plt
from statistics import mean, stdev
import os

def read_instance(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    N, K, M = map(int, lines[0].strip().split())
    E = [list(map(int, lines[i].strip().split())) for i in range(1, M + 1)]
    return N, K, M, np.array(E)

def hard_constraints_violations(timetable, E, M, N):
    violations = 0
    for student in range(M):
        slots_seen = set()
        for exam in range(N):
            if E[student][exam] == 1:
                slot = timetable[exam]
                if slot in slots_seen:
                    violations += 1
                else:
                    slots_seen.add(slot)
    return violations

def get_students_slots(student, timetable, E, N):
    return sorted([timetable[exam] for exam in range(N) if E[student][exam] == 1])

def soft_constraints_violations(timetable, E, M, N, K):
    total_penalty = 0
    for student in range(M):
        slots = get_students_slots(student, timetable, E, N)
        if not slots:
            continue
        
        # Consecutive slots penalty
        for i in range(len(slots) - 1):
            if slots[i + 1] - slots[i] == 1:
                total_penalty += 1
        
        # Long exam gap penalty
        if len(slots) >= 2:
            max_gap = max(slots[i + 1] - slots[i] for i in range(len(slots) - 1))
            if max_gap > K // 2:
                total_penalty += 1
        
        # End cluster penalty
        if all(slot >= K - 2 for slot in slots):
            total_penalty += 1
    
    return total_penalty

def evaluate_fitness(timetable, E, M, N, K):
    hard = hard_constraints_violations(timetable, E, M, N)
    soft = soft_constraints_violations(timetable, E, M, N, K)
    return hard * 1000 + soft

def tournament_selection(population, fitnesses, tournament_size=3):
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = min(indices, key=lambda i: fitnesses[i])
    return population[best_idx].copy()

def crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(timetable, K, mutation_rate=0.05):
    for i in range(len(timetable)):
        if random.random() < mutation_rate:
            timetable[i] = random.randint(0, K - 1)

def local_search(timetable, E, M, N, K, max_iterations=50):
    """Aggressive hill climbing with full neighborhood exploration"""
    current = timetable.copy()
    current_fitness = evaluate_fitness(current, E, M, N, K)
    
    for _ in range(max_iterations):
        best_move = None
        best_fitness = current_fitness
        
        # Try all possible single-exam moves
        for i in range(N):
            old_slot = current[i]
            for new_slot in range(K):
                if new_slot != old_slot:
                    current[i] = new_slot
                    new_fitness = evaluate_fitness(current, E, M, N, K)
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_move = (i, new_slot)
                    current[i] = old_slot
        
        if best_move is None:
            break
        
        # Apply best move
        current[best_move[0]] = best_move[1]
        current_fitness = best_fitness
    
    return current

def run_ga(
    file_path,
    pop_size=100,
    generations=500,
    crossover_rate=0.8,
    mutation_rate=0.05,
    elitism_count=2,
    tournament_size=3,
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    N, K, M, E = read_instance(file_path)
    population = [[random.randint(0, K - 1) for _ in range(N)] for _ in range(pop_size)]
    best_fitness_per_gen = []
    
    stagnation_counter = 0
    adaptive_mutation = mutation_rate

    for gen in range(generations):
        fitnesses = [evaluate_fitness(t, E, M, N, K) for t in population]
        current_best = min(fitnesses)
        best_fitness_per_gen.append(current_best)
        
        # Adaptive mutation: increase if stagnating
        if len(best_fitness_per_gen) > 1 and current_best == best_fitness_per_gen[-2]:
            stagnation_counter += 1
            if stagnation_counter > 20:
                adaptive_mutation = min(0.3, mutation_rate * 2)
        else:
            stagnation_counter = 0
            adaptive_mutation = mutation_rate

        elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:elitism_count]
        new_population = [population[i].copy() for i in elite_indices]
        
        # Apply local search to best solution every 50 generations
        if gen % 50 == 0 and gen > 0:
            best_idx = elite_indices[0]
            new_population[0] = local_search(population[best_idx], E, M, N, K, max_iterations=10)

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            mutate(child1, K, adaptive_mutation)
            mutate(child2, K, adaptive_mutation)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    fitnesses = [evaluate_fitness(t, E, M, N, K) for t in population]
    best_index = fitnesses.index(min(fitnesses))
    best_timetable = population[best_index]
    
    # Final local search refinement
    best_timetable = local_search(best_timetable, E, M, N, K, max_iterations=50)
    best_fitness = evaluate_fitness(best_timetable, E, M, N, K)
    
    return best_timetable, best_fitness_per_gen, E, M, N, K, best_fitness

def run_multiple_trials(file_path, params, num_trials=10):
    """Run GA multiple times with same parameters to get statistics"""
    results = []
    best_fitness_curves = []
    
    print(f"  Running {num_trials} trials...")
    for trial in range(num_trials):
        seed = random.randint(0, 1_000_000)
        best_timetable, fitness_curve, E, M, N, K, best_fitness = run_ga(
            file_path, seed=seed, **params
        )
        hard = hard_constraints_violations(best_timetable, E, M, N)
        soft = soft_constraints_violations(best_timetable, E, M, N, K)
        results.append({
            'timetable': best_timetable,
            'fitness': best_fitness,
            'hard': hard,
            'soft': soft
        })
        best_fitness_curves.append(fitness_curve)
        print(f"    Trial {trial + 1}/{ num_trials}: fitness={best_fitness} (hard={hard}, soft={soft})")
    
    fitnesses = [r['fitness'] for r in results]
    hard_violations = [r['hard'] for r in results]
    soft_violations = [r['soft'] for r in results]
    
    stats = {
        'best_fitness': min(fitnesses),
        'worst_fitness': max(fitnesses),
        'mean_fitness': mean(fitnesses),
        'std_fitness': stdev(fitnesses) if len(fitnesses) > 1 else 0,
        'mean_hard': mean(hard_violations),
        'mean_soft': mean(soft_violations),
        'best_solution': results[fitnesses.index(min(fitnesses))]['timetable'],
        'best_hard': results[fitnesses.index(min(fitnesses))]['hard'],
        'best_soft': results[fitnesses.index(min(fitnesses))]['soft'],
        'fitness_curves': best_fitness_curves
    }
    
    return stats

def format_solution(best_timetable, N):
    """Format solution as exam-to-timeslot mapping"""
    solution_str = "Exam-to-Timeslot Assignment:\n"
    for exam in range(N):
        solution_str += f"  Exam {exam}: Timeslot {best_timetable[exam]}\n"
    return solution_str

def find_best_params(file_path):
    param_grid = [
        {"pop_size": 300, "generations": 400, "crossover_rate": 0.9, "mutation_rate": 0.04, "elitism_count": 5, "tournament_size": 5},
        {"pop_size": 400, "generations": 350, "crossover_rate": 0.85, "mutation_rate": 0.06, "elitism_count": 4, "tournament_size": 6},
        {"pop_size": 250, "generations": 450, "crossover_rate": 0.95, "mutation_rate": 0.03, "elitism_count": 3, "tournament_size": 4},
    ]

    print("\nParameter Comparison Results:")
    param_results = []
    
    for i, params in enumerate(param_grid, 1):
        print(f"\nConfig {i}: {params}")
        result = run_multiple_trials(file_path, params, num_trials=3)
        param_results.append((result['mean_fitness'], params, result))
        print(f"  → Mean fitness: {result['mean_fitness']:.2f}, Best: {result['best_fitness']}, Worst: {result['worst_fitness']}")
    
    # Select best configuration based on mean fitness
    param_results.sort(key=lambda x: x[0])
    best_mean_fitness, best_params, best_stats = param_results[0]
    
    print(f"\n Best configuration selected with mean fitness: {best_mean_fitness:.2f}")
    
    # Get problem dimensions from file
    N, K, M, E = read_instance(file_path)
    
    # Return best params and a result tuple compatible with main()
    return best_params, (best_stats['best_solution'], best_stats['fitness_curves'][0], 
                         E, M, N, K, best_stats['best_fitness'])

def main():
    print("="*80)
    print("EXPERIMENTAL RESULTS FOR EXAM TIMETABLING GENETIC ALGORITHM")
    print("="*80)
    
    # Find all test instances in current directory
    test_files = ['test_case1.txt', 'small-2.txt', 'medium-1.txt', 'tinyexample.txt']
    test_files = [f for f in test_files if os.path.exists(f)]
    
    if not test_files:
        print("No test files found!")
        return
    
    primary_instance = test_files[3]
    
    # PART 1: FIND BEST PARAMETERS
    print("\n" + "="*80)
    print("1. PARAMETER TUNING (Finding Optimal Configuration)")
    print(f"   Primary Instance: {primary_instance}")
    print("="*80)
    
    N, K, M, _ = read_instance(primary_instance)
    print(f"\nProblem: N={N} exams, K={K} timeslots, M={M} students\n")
    
    best_params, best_result = find_best_params(primary_instance)
    best_timetable, fitness_curve, E, M, N, K, best_fitness = best_result
    hard = hard_constraints_violations(best_timetable, E, M, N)
    soft = soft_constraints_violations(best_timetable, E, M, N, K)
    
    print(f"\n BEST PARAMETERS FOUND:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\n  Fitness Achieved: {best_fitness}")
    print(f"  Hard Violations: {hard}")
    print(f"  Soft Violations: {soft}")
    
    # PART 2: PERFORMANCE SUMMARY ON ALL INSTANCES
    print("\n" + "="*80)
    print("2. PERFORMANCE SUMMARY ON ALL TEST INSTANCES")
    print("="*80)
    
    all_instance_results = {}
    
    for test_file in test_files:
        print(f"\nTesting {test_file}...")
        N, K, M, _ = read_instance(test_file)
        print(f"  Problem size: N={N} exams, K={K} timeslots, M={M} students")
        
        seed = random.randint(0, 1_000_000)
        best_timetable, _, E, M, N, K, best_fitness = run_ga(
            test_file, seed=seed, **best_params
        )
        hard = hard_constraints_violations(best_timetable, E, M, N)
        soft = soft_constraints_violations(best_timetable, E, M, N, K)
        
        all_instance_results[test_file] = {
            'N': N, 'K': K, 'M': M,
            'fitness': best_fitness,
            'hard': hard,
            'soft': soft
        }
        print(f"  Fitness: {best_fitness} (Hard: {hard}, Soft: {soft})")
    
    # Print performance table
    print("\n" + "-"*90)
    print(f"{'Test Instance':<20} {'N':<6} {'K':<6} {'M':<6} {'Best Fitness':<15} {'Hard Vio':<12} {'Soft Vio':<12}")
    print("-"*90)
    
    for test_file, results in all_instance_results.items():
        print(f"{test_file:<20} {results['N']:<6} {results['K']:<6} {results['M']:<6} "
              f"{results['fitness']:<15} {results['hard']:<12} {results['soft']:<12}")
    
    # PART 3: CONVERGENCE ANALYSIS
    print("\n" + "="*80)
    print("3. CONVERGENCE ANALYSIS")
    print(f"   ({primary_instance} - Running 5 Independent Trials)")
    print("="*80)
    
    stats = run_multiple_trials(primary_instance, best_params, num_trials=5)
    
    print(f"\n  RESULTS FROM 5 INDEPENDENT RUNS:")
    print(f"    Best Fitness:   {stats['best_fitness']}")
    print(f"    Worst Fitness:  {stats['worst_fitness']}")
    print(f"    Mean Fitness:   {stats['mean_fitness']:.2f} ± {stats['std_fitness']:.2f}")
    print(f"    Mean Hard Violations:  {stats['mean_hard']:.2f}")
    print(f"    Mean Soft Violations:  {stats['mean_soft']:.2f}")
    
    # PART 4: GENERATE CONVERGENCE GRAPHS
    print("\n" + "="*80)
    print("4. GENERATING CONVERGENCE GRAPHS")
    print("="*80)
    
    # Plot convergence curves
    plt.figure(figsize=(12, 6))
    for i, curve in enumerate(stats['fitness_curves']):
        plt.plot(curve, alpha=0.3, linewidth=1, color='steelblue')
    
    avg_curve = np.mean(stats['fitness_curves'], axis=0)
    plt.plot(avg_curve, color='red', linewidth=3, label='Mean Convergence')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title(f'Convergence Over 5 Independent Runs\n{primary_instance}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n Saved: convergence_analysis.png")
    
    # PART 5: BEST SOLUTION EXAMPLE
    print("\n" + "="*80)
    print("5. EXAMPLE BEST SOLUTION FOUND")
    print(f"   ({primary_instance})")
    print("="*80)
    
    best_solution = stats['best_solution']
    N, K, M, _ = read_instance(primary_instance)
    
    print(f"\n{format_solution(best_solution, N)}")
    print(f"Fitness Breakdown:")
    print(f"  Total Fitness: {stats['best_fitness']}")
    print(f"  Hard Constraint Violations: {stats['best_hard']}")
    print(f"  Soft Constraint Violations: {stats['best_soft']}")
    
    plt.show()

if __name__ == "__main__":
    main()