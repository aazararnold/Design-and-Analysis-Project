"""
Simple TSP Demonstration Script
A basic implementation to understand the core concepts of TSP algorithms.
"""

import random
import math
import itertools
import time

def generate_random_cities(n):
    """Generate n random cities with coordinates between 0 and 100"""
    cities = []
    for i in range(n):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        cities.append((x, y))
    return cities

def calculate_distance(city1, city2):
    """Calculate Euclidean distance between two cities"""
    dx = city1[0] - city2[0]
    dy = city1[1] - city2[1]
    return math.sqrt(dx*dx + dy*dy)

def calculate_tour_distance(tour, cities):
    """Calculate total distance of a tour"""
    total_distance = 0
    for i in range(len(tour)):
        current_city = tour[i]
        next_city = tour[(i + 1) % len(tour)]  # Wrap around to start
        total_distance += calculate_distance(cities[current_city], cities[next_city])
    return total_distance

def brute_force_tsp(cities):
    """
    Brute Force TSP Algorithm
    Time Complexity: O(n!)
    Tries all possible permutations to find the optimal solution
    """
    n = len(cities)
    if n > 10:
        print("Too many cities for brute force! (Maximum 10)")
        return None, float('inf')
    
    print(f"Brute Force: Checking {math.factorial(n-1)} possible tours...")
    
    min_distance = float('inf')
    best_tour = None
    
    # Generate all permutations starting from city 0
    other_cities = list(range(1, n))
    
    for perm in itertools.permutations(other_cities):
        tour = [0] + list(perm)  # Always start from city 0
        distance = calculate_tour_distance(tour, cities)
        
        if distance < min_distance:
            min_distance = distance
            best_tour = tour
    
    return best_tour, min_distance

def greedy_nearest_neighbor(cities, start_city=0):
    """
    Greedy Nearest Neighbor Algorithm
    Time Complexity: O(n²)
    Always visits the nearest unvisited city
    """
    n = len(cities)
    unvisited = set(range(n))
    tour = [start_city]
    unvisited.remove(start_city)
    current_city = start_city
    
    print(f"Greedy: Starting from city {start_city}")
    
    while unvisited:
        # Find nearest unvisited city
        nearest_city = min(unvisited, 
                          key=lambda city: calculate_distance(cities[current_city], cities[city]))
        
        print(f"  From city {current_city} -> nearest unvisited city {nearest_city}")
        tour.append(nearest_city)
        unvisited.remove(nearest_city)
        current_city = nearest_city
    
    distance = calculate_tour_distance(tour, cities)
    return tour, distance

def mst_approximation(cities):
    """
    MST-based 2-Approximation Algorithm
    Time Complexity: O(n² log n)
    Uses Minimum Spanning Tree to create a tour with guaranteed quality
    """
    n = len(cities)
    
    print("MST Approximation: Building minimum spanning tree...")
    
    # Step 1: Build distance matrix
    distances = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            distances[i][j] = calculate_distance(cities[i], cities[j])
    
    # Step 2: Find MST using Prim's algorithm
    visited = [False] * n
    key = [float('inf')] * n
    parent = [-1] * n
    key[0] = 0  # Start from vertex 0
    
    mst_edges = []
    
    for _ in range(n):
        # Find minimum key vertex not yet in MST
        u = -1
        for v in range(n):
            if not visited[v] and (u == -1 or key[v] < key[u]):
                u = v
        
        visited[u] = True
        
        if parent[u] != -1:
            mst_edges.append((parent[u], u))
            print(f"  MST edge: {parent[u]} - {u}")
        
        # Update key values of adjacent vertices
        for v in range(n):
            if not visited[v] and distances[u][v] < key[v]:
                key[v] = distances[u][v]
                parent[v] = u
    
    # Step 3: Create adjacency list for MST
    mst_adj = [[] for _ in range(n)]
    for u, v in mst_edges:
        mst_adj[u].append(v)
        mst_adj[v].append(u)
    
    # Step 4: DFS traversal to get tour
    print("MST Approximation: Performing DFS traversal...")
    tour = []
    visited = [False] * n
    
    def dfs(node):
        visited[node] = True
        tour.append(node)
        print(f"  Visiting city {node}")
        
        for neighbor in mst_adj[node]:
            if not visited[neighbor]:
                dfs(neighbor)
    
    dfs(0)  # Start DFS from city 0
    distance = calculate_tour_distance(tour, cities)
    return tour, distance

def print_cities(cities):
    """Print city coordinates"""
    print("Cities:")
    for i, (x, y) in enumerate(cities):
        print(f"  City {i}: ({x:.1f}, {y:.1f})")

def print_tour_result(algorithm_name, tour, distance, execution_time):
    """Print algorithm results"""
    print(f"\n{algorithm_name} Results:")
    print(f"  Tour: {' -> '.join(map(str, tour))} -> {tour[0]}")
    print(f"  Total Distance: {distance:.2f}")
    print(f"  Execution Time: {execution_time*1000:.2f} ms")

def demonstrate_complexity_growth():
    """Demonstrate how execution time grows with problem size"""
    print("\n" + "="*60)
    print("COMPLEXITY GROWTH DEMONSTRATION")
    print("="*60)
    
    sizes = [4, 5, 6, 7, 8]
    
    print("Problem Size | Brute Force Time | Greedy Time | Possible Tours")
    print("-" * 60)
    
    for n in sizes:
        cities = generate_random_cities(n)
        
        # Time brute force
        start_time = time.time()
        _, _ = brute_force_tsp(cities)
        bf_time = time.time() - start_time
        
        # Time greedy
        start_time = time.time()
        _, _ = greedy_nearest_neighbor(cities)
        greedy_time = time.time() - start_time
        
        possible_tours = math.factorial(n-1)
        
        print(f"     {n:2d}      |    {bf_time*1000:8.2f} ms    |  {greedy_time*1000:6.2f} ms  | {possible_tours:,}")

def main():
    """Main demonstration function"""
    print("="*60)
    print("TRAVELING SALESMAN PROBLEM (TSP) DEMONSTRATION")
    print("NP-Hard Problem Analysis for DAA Course")
    print("="*60)
    
    # Generate a small set of cities for demonstration
    n_cities = 6
    print(f"\nGenerating {n_cities} random cities...")
    cities = generate_random_cities(n_cities)
    print_cities(cities)
    
    print(f"\nTotal possible tours: {math.factorial(n_cities-1):,}")
    print("(This is why TSP is computationally challenging!)")
    
    algorithms = [
        ("Brute Force (Exact)", brute_force_tsp),
        ("Greedy Nearest Neighbor (Heuristic)", greedy_nearest_neighbor),
        ("MST Approximation (2-Approximation)", mst_approximation)
    ]
    
    results = []
    
    print("\n" + "="*60)
    print("RUNNING ALGORITHMS")
    print("="*60)
    
    for name, algorithm in algorithms:
        print(f"\n{name}:")
        print("-" * len(name))
        
        start_time = time.time()
        tour, distance = algorithm(cities)
        end_time = time.time()
        
        execution_time = end_time - start_time
        results.append((name, tour, distance, execution_time))
        
        if tour is not None:
            print_tour_result(name, tour, distance, execution_time)
        else:
            print("  Algorithm could not complete (problem too large)")
    
    # Compare results
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON")
    print("="*60)
    
    print("Algorithm                    | Distance | Time (ms) | Quality")
    print("-" * 65)
    
    optimal_distance = min(result[2] for result in results if result[1] is not None)
    
    for name, tour, distance, exec_time in results:
        if tour is not None:
            quality_ratio = distance / optimal_distance
            quality = "Optimal" if quality_ratio < 1.01 else f"{quality_ratio:.2f}x optimal"
            print(f"{name:28} | {distance:8.2f} | {exec_time*1000:8.2f} | {quality}")
    
    # Demonstrate complexity growth
    demonstrate_complexity_growth()
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("1. Brute force guarantees optimal solution but has factorial time complexity")
    print("2. Greedy algorithm is fast (quadratic) but may not find optimal solution")
    print("3. MST approximation provides quality guarantee (≤ 2x optimal)")
    print("4. For large problems, heuristics and approximations are essential")
    print("5. This demonstrates why P ≠ NP is such an important question!")

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    main()
