import random
from random import randint, shuffle
import matplotlib.pyplot as plt
import time
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SelectionMethod(Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"

class CrossoverMethod(Enum):
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    PMX = "pmx"  # Partially Mapped Crossover for permutations

@dataclass
class GeneticAlgorithmConfig:
    population_size: int = 50
    board_size: int = 8
    mutation_rate: float = 0.15
    crossover_rate: float = 0.85
    max_generations: int = 1000
    elitism_count: int = 2
    tournament_size: int = 3
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.PMX
    early_stopping_patience: int = 50
    parallel_evaluation: bool = True
    max_workers: int = 4

class AdvancedNQueensSolver:
    def __init__(self, config: GeneticAlgorithmConfig):
        self.config = config
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generation_count = 0
        self.stagnation_count = 0
        self.best_fitness = float('inf')
        
    def initialize_population(self) -> List[List[int]]:
        """Initialize population with diverse strategies"""
        population = []
        
        for _ in range(self.config.population_size):
            # Use different initialization strategies for diversity
            strategy = randint(0, 2)
            if strategy == 0:
                # Random initialization
                individual = [randint(0, self.config.board_size - 1) for _ in range(self.config.board_size)]
            elif strategy == 1:
                # Diagonal initialization (often good starting point)
                individual = [i for i in range(self.config.board_size)]
                shuffle(individual)
            else:
                # Column-wise unique positions
                individual = list(range(self.config.board_size))
                shuffle(individual)
            
            population.append(individual)
        
        return population
    
    def calculate_fitness(self, individual: List[int]) -> int:
        """Calculate conflicts with optimized diagonal checking"""
        conflicts = 0
        n = len(individual)
        
        # Check for row conflicts (queens in same row)
        row_count = [0] * n
        for row in individual:
            row_count[row] += 1
        conflicts += sum(count - 1 for count in row_count if count > 1)
        
        # Check diagonal conflicts using mathematical properties
        # Main diagonal: row - col = constant
        main_diag = [0] * (2 * n)
        # Anti-diagonal: row + col = constant  
        anti_diag = [0] * (2 * n)
        
        for col, row in enumerate(individual):
            main_diag[row - col + n] += 1
            anti_diag[row + col] += 1
        
        conflicts += sum(count - 1 for count in main_diag if count > 1)
        conflicts += sum(count - 1 for count in anti_diag if count > 1)
        
        return conflicts
    
    def evaluate_population(self, population: List[List[int]]) -> List[Tuple[List[int], int]]:
        """Evaluate population fitness, optionally in parallel"""
        if self.config.parallel_evaluation and self.config.board_size >= 8:
            return self._evaluate_parallel(population)
        else:
            return self._evaluate_sequential(population)
    
    def _evaluate_sequential(self, population: List[List[int]]) -> List[Tuple[List[int], int]]:
        return [(ind, self.calculate_fitness(ind)) for ind in population]
    
    def _evaluate_parallel(self, population: List[List[int]]) -> List[Tuple[List[int], int]]:
        """Parallel fitness evaluation"""
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_individual = {
                executor.submit(self.calculate_fitness, ind): ind 
                for ind in population
            }
            
            results = []
            for future in as_completed(future_to_individual):
                individual = future_to_individual[future]
                try:
                    fitness = future.result()
                    results.append((individual, fitness))
                except Exception as exc:
                    logger.error(f"Fitness calculation generated an exception: {exc}")
                    # Fallback: assign worst fitness
                    results.append((individual, self.config.board_size * 3))
        
        return results
    
    def selection(self, evaluated_population: List[Tuple[List[int], int]]) -> List[List[int]]:
        """Select parents using specified method"""
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(evaluated_population)
        elif self.config.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection(evaluated_population)
        elif self.config.selection_method == SelectionMethod.RANK:
            return self._rank_selection(evaluated_population)
        else:
            return self._tournament_selection(evaluated_population)
    
    def _tournament_selection(self, evaluated_population: List[Tuple[List[int], int]]) -> List[List[int]]:
        """Tournament selection"""
        selected = []
        
        # Always keep the best individual (elitism)
        sorted_population = sorted(evaluated_population, key=lambda x: x[1])
        for i in range(min(self.config.elitism_count, len(sorted_population))):
            selected.append(sorted_population[i][0])
        
        # Tournament selection for the rest
        while len(selected) < self.config.population_size:
            tournament = random.sample(evaluated_population, self.config.tournament_size)
            winner = min(tournament, key=lambda x: x[1])
            selected.append(winner[0])
        
        return selected
    
    def _roulette_selection(self, evaluated_population: List[Tuple[List[int], int]]) -> List[List[int]]:
        """Roulette wheel selection (minimization problem)"""
        # Convert to maximization problem for roulette
        max_fitness = max(fitness for _, fitness in evaluated_population) + 1
        fitness_values = [max_fitness - fitness for _, fitness in evaluated_population]
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return [ind for ind, _ in evaluated_population]
        
        probabilities = [f / total_fitness for f in fitness_values]
        selected_indices = np.random.choice(
            len(evaluated_population), 
            size=self.config.population_size - self.config.elitism_count,
            p=probabilities
        )
        
        selected = [evaluated_population[i][0] for i in selected_indices]
        
        # Add elites
        sorted_population = sorted(evaluated_population, key=lambda x: x[1])
        for i in range(min(self.config.elitism_count, len(sorted_population))):
            selected.append(sorted_population[i][0])
        
        return selected
    
    def _rank_selection(self, evaluated_population: List[Tuple[List[int], int]]) -> List[List[int]]:
        """Rank-based selection"""
        sorted_population = sorted(evaluated_population, key=lambda x: x[1])
        ranks = list(range(1, len(sorted_population) + 1))
        total_rank = sum(ranks)
        probabilities = [r / total_rank for r in ranks]
        
        selected_indices = np.random.choice(
            len(sorted_population),
            size=self.config.population_size - self.config.elitism_count,
            p=probabilities
        )
        
        selected = [sorted_population[i][0] for i in selected_indices]
        
        # Add elites
        for i in range(min(self.config.elitism_count, len(sorted_population))):
            selected.append(sorted_population[i][0])
        
        return selected
    
    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Crossover using specified method"""
        if random.random() > self.config.crossover_rate:
            return parent1[:], parent2[:]
        
        if self.config.crossover_method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self.config.crossover_method == CrossoverMethod.PMX:
            return self._pmx_crossover(parent1, parent2)
        else:
            return self._pmx_crossover(parent1, parent2)
    
    def _single_point_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        point = randint(1, self.config.board_size - 2)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def _two_point_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        point1 = randint(0, self.config.board_size - 2)
        point2 = randint(point1 + 1, self.config.board_size - 1)
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
        return child1, child2
    
    def _uniform_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        child1, child2 = [], []
        for i in range(self.config.board_size):
            if random.random() < 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return child1, child2
    
    def _pmx_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Partially Mapped Crossover - good for permutation problems"""
        size = len(parent1)
        point1 = randint(0, size - 2)
        point2 = randint(point1 + 1, size - 1)
        
        def pmx_child(p1, p2):
            child = [None] * size
            
            # Copy segment between points
            child[point1:point2] = p1[point1:point2]
            
            # Fill remaining positions
            for i in list(range(0, point1)) + list(range(point2, size)):
                candidate = p2[i]
                while candidate in child[point1:point2]:
                    idx = p1[point1:point2].index(candidate)
                    candidate = p2[point1 + idx]
                child[i] = candidate
            
            return child
        
        child1 = pmx_child(parent1, parent2)
        child2 = pmx_child(parent2, parent1)
        
        return child1, child2
    
    def mutation(self, individual: List[int]) -> List[int]:
        """Apply mutation with different strategies"""
        if random.random() > self.config.mutation_rate:
            return individual
        
        mutated = individual[:]
        mutation_type = random.choice(["swap", "scramble", "inversion", "random_reset"])
        
        if mutation_type == "swap":
            # Swap two random positions
            idx1, idx2 = random.sample(range(self.config.board_size), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        elif mutation_type == "scramble":
            # Scramble a random segment
            start = randint(0, self.config.board_size - 2)
            end = randint(start + 1, self.config.board_size - 1)
            segment = mutated[start:end]
            shuffle(segment)
            mutated[start:end] = segment
        
        elif mutation_type == "inversion":
            # Invert a random segment
            start = randint(0, self.config.board_size - 2)
            end = randint(start + 1, self.config.board_size - 1)
            mutated[start:end] = reversed(mutated[start:end])
        
        elif mutation_type == "random_reset":
            # Reset a random position
            idx = randint(0, self.config.board_size - 1)
            mutated[idx] = randint(0, self.config.board_size - 1)
        
        return mutated
    
    def create_new_generation(self, parents: List[List[int]]) -> List[List[int]]:
        """Create new generation through crossover and mutation"""
        new_population = []
        shuffle(parents)
        
        # Elitism: carry forward best individuals
        evaluated_parents = self.evaluate_population(parents)
        sorted_parents = sorted(evaluated_parents, key=lambda x: x[1])
        for i in range(min(self.config.elitism_count, len(sorted_parents))):
            new_population.append(sorted_parents[i][0])
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.config.population_size:
            if len(parents) >= 2:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.append(self.mutation(child1))
                if len(new_population) < self.config.population_size:
                    new_population.append(self.mutation(child2))
            else:
                # Fallback: clone and mutate
                parent = random.choice(parents)
                new_population.append(self.mutation(parent[:]))
        
        return new_population[:self.config.population_size]
    
    def solve(self) -> Tuple[Optional[List[int]], int, float]:
        """Main solving method"""
        start_time = time.time()
        
        # Initialize population
        self.population = self.initialize_population()
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.generation_count = 0
        self.stagnation_count = 0
        self.best_fitness = float('inf')
        best_solution = None
        
        logger.info(f"Starting Genetic Algorithm for {self.config.board_size}-Queens Problem")
        logger.info(f"Population: {self.config.population_size}, Generations: {self.config.max_generations}")
        
        for generation in range(self.config.max_generations):
            self.generation_count = generation + 1
            
            # Evaluate population
            evaluated_population = self.evaluate_population(self.population)
            
            # Track statistics
            fitness_values = [fitness for _, fitness in evaluated_population]
            current_best_fitness = min(fitness_values)
            current_avg_fitness = sum(fitness_values) / len(fitness_values)
            
            self.best_fitness_history.append(current_best_fitness)
            self.avg_fitness_history.append(current_avg_fitness)
            
            # Check for improvement
            if current_best_fitness < self.best_fitness:
                self.best_fitness = current_best_fitness
                self.stagnation_count = 0
                # Find the best solution
                for individual, fitness in evaluated_population:
                    if fitness == current_best_fitness:
                        best_solution = individual
                        break
                
                logger.info(f"Generation {generation + 1}: Best fitness = {current_best_fitness}")
                
                # Check for solution
                if current_best_fitness == 0:
                    logger.info(f"Solution found at generation {generation + 1}!")
                    break
            else:
                self.stagnation_count += 1
            
            # Early stopping
            if self.stagnation_count >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at generation {generation + 1} due to stagnation")
                break
            
            # Selection
            parents = self.selection(evaluated_population)
            
            # Create new generation
            self.population = self.create_new_generation(parents)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return best_solution, self.generation_count, execution_time
    
    def plot_solution(self, solution: List[int], title: str = "N-Queens Solution"):
        """Visualize the solution"""
        if solution is None:
            logger.warning("No solution to plot")
            return
        
        n = len(solution)
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create chessboard
        for i in range(n + 1):
            ax.plot([0, n], [i, i], 'k-', lw=2)
            ax.plot([i, i], [0, n], 'k-', lw=2)
        
        # Color the chessboard
        for i in range(n):
            for j in range(n):
                color = 'white' if (i + j) % 2 == 0 else 'lightgray'
                ax.fill_between([i, i+1], j, j+1, color=color, alpha=0.7)
        
        # Place queens
        for col, row in enumerate(solution):
            ax.text(col + 0.5, row + 0.5, '♛', fontsize=24, 
                   ha='center', va='center', color='red')
        
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}\n{self.config.board_size}-Queens Solution", fontsize=14)
        
        plt.tight_layout()
        plt.show()
    
    def plot_convergence(self):
        """Plot fitness convergence over generations"""
        plt.figure(figsize=(10, 6))
        generations = range(1, len(self.best_fitness_history) + 1)
        
        plt.plot(generations, self.best_fitness_history, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(generations, self.avg_fitness_history, 'r--', label='Average Fitness', linewidth=2)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Conflicts)')
        plt.title('Genetic Algorithm Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Advanced N-Queens Genetic Algorithm Solver')
    parser.add_argument('--size', type=int, default=8, help='Board size (N)')
    parser.add_argument('--population', type=int, default=50, help='Population size')
    parser.add_argument('--generations', type=int, default=1000, help='Maximum generations')
    parser.add_argument('--mutation-rate', type=float, default=0.15, help='Mutation rate')
    parser.add_argument('--crossover-rate', type=float, default=0.85, help='Crossover rate')
    parser.add_argument('--selection', choices=['tournament', 'roulette', 'rank'], 
                       default='tournament', help='Selection method')
    parser.add_argument('--crossover', choices=['single_point', 'two_point', 'uniform', 'pmx'], 
                       default='pmx', help='Crossover method')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel evaluation')
    
    args = parser.parse_args()
    
    # Create configuration
    config = GeneticAlgorithmConfig(
        population_size=args.population,
        board_size=args.size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        max_generations=args.generations,
        selection_method=SelectionMethod(args.selection),
        crossover_method=CrossoverMethod(args.crossover),
        parallel_evaluation=not args.no_parallel
    )
    
    # Solve the problem
    solver = AdvancedNQueensSolver(config)
    solution, generations, execution_time = solver.solve()
    
    # Display results
    print("\n" + "="*50)
    print("N-QUEENS GENETIC ALGORITHM RESULTS")
    print("="*50)
    print(f"Board Size: {config.board_size}")
    print(f"Solution Found: {solution is not None}")
    print(f"Generations: {generations}")
    print(f"Execution Time: {execution_time:.2f} seconds")
    
    if solution:
        print(f"Solution: {solution}")
        conflicts = solver.calculate_fitness(solution)
        print(f"Conflicts: {conflicts}")
        
        # Visualize
        solver.plot_solution(solution)
        solver.plot_convergence()
    else:
        print("No solution found within the given constraints")
        solver.plot_convergence()

if __name__ == "__main__":
    main()