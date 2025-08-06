"""
Standalone Performance Testing Script for TSP Algorithms
This script runs comprehensive performance tests and generates analysis reports.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import math
import itertools

class TSPPerformanceTester:
    def __init__(self):
        pass
    
    @staticmethod
    def generate_cities(n):
        """Generate n random cities"""
        return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n)]
    
    @staticmethod
    def calculate_distance(city1, city2):
        """Calculate Euclidean distance between two cities"""
        return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
    
    @staticmethod
    def calculate_tour_distance(tour, cities):
        """Calculate total distance of a tour"""
        total = 0
        for i in range(len(tour)):
            current = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            total += TSPPerformanceTester.calculate_distance(cities[current], cities[next_city])
        return total
    
    @staticmethod
    def brute_force_tsp(cities):
        """Brute force TSP solver"""
        n = len(cities)
        if n > 10:
            return None, float('inf')
        
        min_distance = float('inf')
        best_tour = None
        
        for perm in itertools.permutations(range(1, n)):
            tour = [0] + list(perm)
            distance = TSPPerformanceTester.calculate_tour_distance(tour, cities)
            if distance < min_distance:
                min_distance = distance
                best_tour = tour
        
        return best_tour, min_distance
    
    @staticmethod
    def greedy_tsp(cities):
        """Greedy nearest neighbor TSP solver"""
        n = len(cities)
        unvisited = set(range(1, n))
        tour = [0]
        current = 0
        
        while unvisited:
            nearest = min(unvisited, 
                         key=lambda city: TSPPerformanceTester.calculate_distance(cities[current], cities[city]))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        distance = TSPPerformanceTester.calculate_tour_distance(tour, cities)
        return tour, distance
    
    @staticmethod
    def mst_approximation_tsp(cities):
        """MST-based approximation TSP solver"""
        n = len(cities)
        
        # Build distance matrix
        distances = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                distances[i][j] = TSPPerformanceTester.calculate_distance(cities[i], cities[j])
        
        # Prim's MST algorithm
        visited = [False] * n
        key = [float('inf')] * n
        parent = [-1] * n
        key[0] = 0
        
        mst_edges = []
        for _ in range(n):
            u = -1
            for v in range(n):
                if not visited[v] and (u == -1 or key[v] < key[u]):
                    u = v
            
            visited[u] = True
            if parent[u] != -1:
                mst_edges.append((parent[u], u))
            
            for v in range(n):
                if not visited[v] and distances[u][v] < key[v]:
                    key[v] = distances[u][v]
                    parent[v] = u
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in mst_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # DFS traversal
        tour = []
        visited = [False] * n
        
        def dfs(node):
            visited[node] = True
            tour.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        dfs(0)
        distance = TSPPerformanceTester.calculate_tour_distance(tour, cities)
        return tour, distance
    
    @staticmethod
    def simulated_annealing_tsp(cities):
        """Simulated Annealing TSP solver"""
        n = len(cities)
        
        # Generate initial random solution
        current_solution = list(range(n))
        random.shuffle(current_solution)
        current_cost = TSPPerformanceTester.calculate_tour_distance(current_solution, cities)
        
        best_solution = current_solution[:]
        best_cost = current_cost
        
        # SA parameters
        initial_temp = 1000
        cooling_rate = 0.995
        min_temp = 1
        max_iterations = 5000
        
        temperature = initial_temp
        iteration = 0
        
        while temperature > min_temp and iteration < max_iterations:
            # Generate neighbor by swapping two random cities
            new_solution = current_solution[:]
            i, j = random.sample(range(n), 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            
            new_cost = TSPPerformanceTester.calculate_tour_distance(new_solution, cities)
            
            # Calculate acceptance probability
            if new_cost < current_cost:
                # Always accept better solution
                current_solution = new_solution
                current_cost = new_cost
                
                # Update best solution if necessary
                if new_cost < best_cost:
                    best_solution = new_solution[:]
                    best_cost = new_cost
            else:
                # Accept worse solution with probability
                delta = new_cost - current_cost
                probability = math.exp(-delta / temperature)
                
                if random.random() < probability:
                    current_solution = new_solution
                    current_cost = new_cost
        
        # Cool down
        temperature *= cooling_rate
        iteration += 1
        
        return best_solution, best_cost
    
    def run_performance_test(self):
        """Run comprehensive performance test"""
        print("Running TSP Performance Analysis...")
        print("=" * 50)
        
        # Test parameters
        city_counts = [4, 5, 6, 7, 8, 10, 12, 15, 20]
        num_trials = 5
        
        # Results storage
        results = {
            'brute_force': {'sizes': [], 'times': [], 'distances': []},
            'greedy': {'sizes': [], 'times': [], 'distances': []},
            'mst': {'sizes': [], 'times': [], 'distances': []},
            'simulated_annealing': {'sizes': [], 'times': [], 'distances': []}
        }
        
        for n in city_counts:
            print(f"\nTesting with {n} cities...")
            
            # Average over multiple trials
            bf_times, bf_distances = [], []
            greedy_times, greedy_distances = [], []
            mst_times, mst_distances = [], []
            sa_times, sa_distances = [], []
            
            for trial in range(num_trials):
                cities = self.generate_cities(n)
                
                # Test Brute Force (only for small n)
                if n <= 8:
                    start_time = time.time()
                    tour, distance = self.brute_force_tsp(cities)
                    end_time = time.time()
                    if tour is not None:
                        bf_times.append((end_time - start_time) * 1000)
                        bf_distances.append(distance)
                
                # Test Greedy
                start_time = time.time()
                tour, distance = self.greedy_tsp(cities)
                end_time = time.time()
                greedy_times.append((end_time - start_time) * 1000)
                greedy_distances.append(distance)
                
                # Test MST
                start_time = time.time()
                tour, distance = self.mst_approximation_tsp(cities)
                end_time = time.time()
                mst_times.append((end_time - start_time) * 1000)
                mst_distances.append(distance)
            
                # Test Simulated Annealing
                start_time = time.time()
                tour, distance = self.simulated_annealing_tsp(cities)
                end_time = time.time()
                sa_times.append((end_time - start_time) * 1000)
                sa_distances.append(distance)
            
            # Store average results
            if bf_times:
                results['brute_force']['sizes'].append(n)
                results['brute_force']['times'].append(np.mean(bf_times))
                results['brute_force']['distances'].append(np.mean(bf_distances))
            
            results['greedy']['sizes'].append(n)
            results['greedy']['times'].append(np.mean(greedy_times))
            results['greedy']['distances'].append(np.mean(greedy_distances))
            
            results['mst']['sizes'].append(n)
            results['mst']['times'].append(np.mean(mst_times))
            results['mst']['distances'].append(np.mean(mst_distances))
            
            results['simulated_annealing']['sizes'].append(n)
            results['simulated_annealing']['times'].append(np.mean(sa_times))
            results['simulated_annealing']['distances'].append(np.mean(sa_distances))

            print(f"  Greedy: {np.mean(greedy_times):.2f}ms, Distance: {np.mean(greedy_distances):.2f}")
            print(f"  MST: {np.mean(mst_times):.2f}ms, Distance: {np.mean(mst_distances):.2f}")
            print(f"  Simulated Annealing: {np.mean(sa_times):.2f}ms, Distance: {np.mean(sa_distances):.2f}")
            if bf_times:
                print(f"  Brute Force: {np.mean(bf_times):.2f}ms, Distance: {np.mean(bf_distances):.2f}")
        
        # Generate plots
        self.generate_performance_plots(results)
        
        # Print summary
        self.print_summary(results)
    
    def generate_performance_plots(self, results):
        """Generate performance analysis plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Execution Time Comparison
        if results['brute_force']['sizes']:
            ax1.plot(results['brute_force']['sizes'], results['brute_force']['times'], 
                    'r-o', label='Brute Force O(n!)', linewidth=2, markersize=6)
        
        ax1.plot(results['greedy']['sizes'], results['greedy']['times'], 
                'g-s', label='Greedy O(n²)', linewidth=2, markersize=6)
        ax1.plot(results['mst']['sizes'], results['mst']['times'], 
                'b-^', label='MST O(n²logn)', linewidth=2, markersize=6)
        ax1.plot(results['simulated_annealing']['sizes'], results['simulated_annealing']['times'], 
                'm-d', label='Simulated Annealing O(k×n)', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Number of Cities')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Time Complexity Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Solution Quality Comparison
        if results['brute_force']['sizes']:
            ax2.plot(results['brute_force']['sizes'], results['brute_force']['distances'], 
                    'r-o', label='Brute Force (Optimal)', linewidth=2, markersize=6)
        
        ax2.plot(results['greedy']['sizes'], results['greedy']['distances'], 
                'g-s', label='Greedy Heuristic', linewidth=2, markersize=6)
        ax2.plot(results['mst']['sizes'], results['mst']['distances'], 
                'b-^', label='MST Approximation', linewidth=2, markersize=6)
        ax2.plot(results['simulated_annealing']['sizes'], results['simulated_annealing']['distances'], 
                'm-d', label='Simulated Annealing', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Number of Cities')
        ax2.set_ylabel('Tour Distance')
        ax2.set_title('Solution Quality Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Theoretical Complexity Growth
        n_values = range(3, 11)
        factorial_growth = [math.factorial(n) / 1000000 for n in n_values]  # Normalized
        quadratic_growth = [n*n for n in n_values]
        nlogn_growth = [n * math.log(n) for n in n_values]
        
        ax3.plot(n_values, factorial_growth, 'r--', label='O(n!) Factorial', linewidth=2)
        ax3.plot(n_values, quadratic_growth, 'g--', label='O(n²) Quadratic', linewidth=2)
        ax3.plot(n_values, nlogn_growth, 'b--', label='O(n log n)', linewidth=2)
        
        ax3.set_xlabel('Input Size (n)')
        ax3.set_ylabel('Relative Operations')
        ax3.set_title('Theoretical Complexity Growth')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Plot 4: Algorithm Efficiency (Time vs Quality)
        if results['brute_force']['sizes']:
            ax4.scatter(results['brute_force']['times'], results['brute_force']['distances'], 
                       c='red', s=100, alpha=0.7, label='Brute Force')
        
        ax4.scatter(results['greedy']['times'], results['greedy']['distances'], 
                   c='green', s=100, alpha=0.7, label='Greedy')
        ax4.scatter(results['mst']['times'], results['mst']['distances'], 
                   c='blue', s=100, alpha=0.7, label='MST')
        ax4.scatter(results['simulated_annealing']['times'], results['simulated_annealing']['distances'], 
                   c='magenta', s=100, alpha=0.7, label='Simulated Annealing')
        
        ax4.set_xlabel('Execution Time (ms)')
        ax4.set_ylabel('Tour Distance')
        ax4.set_title('Algorithm Efficiency (Lower-Left is Better)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tsp_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nPerformance analysis plots saved as 'tsp_performance_analysis.png'")
    
    def print_summary(self, results):
        """Print performance analysis summary"""
        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("=" * 60)
        
        print("\n1. TIME COMPLEXITY OBSERVATIONS:")
        print("   • Brute Force: Exponential growth O(n!) - impractical beyond 8-10 cities")
        print("   • Greedy: Quadratic growth O(n²) - scales well for large instances")
        print("   • MST Approximation: O(n² log n) - slightly slower than greedy but better quality")
        
        print("\n2. SOLUTION QUALITY:")
        print("   • Brute Force: Always finds optimal solution (when feasible)")
        print("   • MST Approximation: Guarantees solution ≤ 2 × optimal")
        print("   • Greedy: No quality guarantee, but often performs reasonably well")
        
        print("\n3. PRACTICAL RECOMMENDATIONS:")
        print("   • Small instances (n ≤ 8): Use Brute Force for optimal solutions")
        print("   • Medium instances (8 < n ≤ 50): Use MST Approximation for quality guarantee")
        print("   • Large instances (n > 50): Use Greedy for speed, or advanced metaheuristics")
        
        print("\n4. ALGORITHM COMPARISON TABLE:")
        print("   ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐")
        print("   │ Algorithm       │ Time Complexity │ Space Complexity│ Solution Quality│")
        print("   ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤")
        print("   │ Brute Force     │ O(n!)           │ O(n)            │ Optimal         │")
        print("   │ Greedy NN       │ O(n²)           │ O(n)            │ Heuristic       │")
        print("   │ MST Approximation│ O(n² log n)     │ O(n²)           │ ≤ 2 × Optimal   │")
        print("   │ Genetic Algorithm│ O(g × p × n)    │ O(p × n)        │ Heuristic       │")
        print("   └─────────────────┴─────────────────┴─────────────────┴─────────────────┘")
        
        print("\n5. KEY INSIGHTS:")
        print("   • NP-Hard problems require trade-offs between optimality and efficiency")
        print("   • Approximation algorithms provide theoretical guarantees")
        print("   • Heuristics are essential for real-world large-scale problems")
        print("   • Algorithm choice depends on problem size and quality requirements")

def main():
    """Main function to run performance testing"""
    print("TSP Performance Testing Script")
    print("==============================")
    print("This script will test different TSP algorithms and generate analysis.")
    print("The test may take a few minutes to complete...\n")
    
    tester = TSPPerformanceTester()
    tester.run_performance_test()
    
    print("\nPerformance testing completed!")
    print("Check the generated plots and analysis above.")

if __name__ == "__main__":
    main()