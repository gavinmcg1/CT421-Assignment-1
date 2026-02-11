# Exam Timetabling Genetic Algorithm

A genetic algorithm implementation for solving the exam timetabling problem, minimizing scheduling conflicts and optimizing student exam distribution.

## Problem Definition

**Input:** N exams, K timeslots, M students with enrollment matrix  
**Objective:** Assign each exam to a timeslot while:
- **Hard constraints:** No student has multiple exams at the same time
- **Soft constraints:** Minimize consecutive exams, long gaps, and end clustering

## Code Structure

### Core Algorithm Components
- `read_instance()` - Load problem instances from file
- `evaluate_fitness()` - Calculate solution quality (hard * 1000 + soft)
- `hard_constraints_violations()` - Count scheduling conflicts
- `soft_constraints_violations()` - Measure schedule quality penalties

### Genetic Algorithm Operators
- `tournament_selection()` - Parent selection mechanism
- `crossover()` - Single-point crossover for offspring generation
- `mutate()` - Random mutation for genetic diversity
- `local_search()` - Hill climbing refinement

### Main Functions
- `run_ga()` - Main genetic algorithm with adaptive mutation and elitism
- `run_multiple_trials()` - Statistical analysis across multiple runs
- `find_best_params()` - Parameter tuning on test configurations
- `main()` - Full experimental workflow

## Algorithm Features

- **Adaptive mutation:** Increases mutation rate when stagnating
- **Elitism:** Preserves best solutions across generations
- **Hybrid approach:** Combines GA with periodic local search
- **Statistical evaluation:** Multiple trials with convergence analysis

## Usage

```bash
python "Exam Timetabling.py"
```

The program will:
1. Tune parameters on primary test instance
2. Evaluate performance across all test files
3. Run convergence analysis with 5 independent trials
4. Generate convergence visualization graph
5. Display best solution found

## Output

- Console output with detailed results and statistics
- `convergence_analysis.png` - Convergence plot across trials

## Test Instances

Place `.txt` files in the same directory with format:
```
N K M
[M lines of enrollment matrix, N columns each]
```
