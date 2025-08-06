"""
Traveling Salesman Problem (TSP) - NP-Hard Problem Analysis
Design and Analysis of Algorithms Course Project

This project implements and compares different algorithms for solving TSP:
1. Brute Force (exact solution)
2. Greedy Nearest Neighbor (heuristic)
3. MST Approximation (approximation algorithm)
4. Genetic Algorithm (metaheuristic)
5. Simulated Annealing (metaheuristic)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import numpy as np
import random
import itertools
import time
import math

class TSPSolver:
    def __init__(self):
        self.cities = []
        self.distances = []
        
    def set_cities(self, cities):
        """Set the cities and calculate distance matrix"""
        self.cities = cities
        n = len(cities)
        self.distances = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = cities[i][0] - cities[j][0]
                    dy = cities[i][1] - cities[j][1]
                    self.distances[i][j] = math.sqrt(dx*dx + dy*dy)
    
    def calculate_tour_distance(self, tour):
        """Calculate total distance of a tour"""
        total_distance = 0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            total_distance += self.distances[current_city][next_city]
        return total_distance
    
    def brute_force_tsp(self):
        """Brute force algorithm - tries all permutations"""
        n = len(self.cities)
        if n > 10:
            return None, float('inf'), "Too many cities for brute force (max 10)"
        
        min_distance = float('inf')
        best_tour = None
        
        # Generate all permutations starting from city 0
        cities_to_permute = list(range(1, n))
        
        for perm in itertools.permutations(cities_to_permute):
            tour = [0] + list(perm)
            distance = self.calculate_tour_distance(tour)
            
            if distance < min_distance:
                min_distance = distance
                best_tour = tour
        
        return best_tour, min_distance, "Optimal solution found"
    
    def greedy_nearest_neighbor_animated(self, start_city=0):
        """Greedy nearest neighbor with step-by-step animation data"""
        n = len(self.cities)
        unvisited = set(range(n))
        tour = [start_city]
        unvisited.remove(start_city)
        current_city = start_city
        
        # Store animation steps
        animation_steps = []
        animation_steps.append({
            'current_city': current_city,
            'tour': tour[:],
            'unvisited': unvisited.copy(),
            'message': f"Starting at city {start_city}"
        })
        
        while unvisited:
            # Find nearest unvisited city
            nearest_city = min(unvisited, key=lambda city: self.distances[current_city][city])
            nearest_distance = self.distances[current_city][nearest_city]
            
            tour.append(nearest_city)
            unvisited.remove(nearest_city)
            
            animation_steps.append({
                'current_city': current_city,
                'next_city': nearest_city,
                'tour': tour[:],
                'unvisited': unvisited.copy(),
                'distance': nearest_distance,
                'message': f"From city {current_city} → nearest city {nearest_city} (distance: {nearest_distance:.2f})"
            })
            
            current_city = nearest_city
        
        # Add final step - return to start
        final_distance = self.distances[current_city][start_city]
        animation_steps.append({
            'current_city': current_city,
            'next_city': start_city,
            'tour': tour + [start_city],
            'unvisited': set(),
            'distance': final_distance,
            'message': f"Return to start: city {current_city} → city {start_city} (distance: {final_distance:.2f})"
        })
        
        total_distance = self.calculate_tour_distance(tour)
        return tour, total_distance, "Heuristic solution", animation_steps
    
    def greedy_nearest_neighbor(self, start_city=0):
        """Regular greedy nearest neighbor without animation"""
        tour, distance, message, _ = self.greedy_nearest_neighbor_animated(start_city)
        return tour, distance, message
    
    def mst_approximation(self):
        """MST-based 2-approximation algorithm"""
        n = len(self.cities)
        
        # Step 1: Find Minimum Spanning Tree using Prim's algorithm
        visited = [False] * n
        key = [float('inf')] * n
        parent = [-1] * n
        key[0] = 0
        
        mst_edges = []
        
        for _ in range(n):
            # Find minimum key vertex not yet visited
            u = -1
            for v in range(n):
                if not visited[v] and (u == -1 or key[v] < key[u]):
                    u = v
            
            visited[u] = True
            
            if parent[u] != -1:
                mst_edges.append((parent[u], u))
            
            # Update key values of adjacent vertices
            for v in range(n):
                if not visited[v] and self.distances[u][v] < key[v]:
                    key[v] = self.distances[u][v]
                    parent[v] = u
        
        # Step 2: Create adjacency list for MST
        mst_adj = [[] for _ in range(n)]
        for u, v in mst_edges:
            mst_adj[u].append(v)
            mst_adj[v].append(u)
        
        # Step 3: DFS traversal to get tour
        tour = []
        visited = [False] * n
        
        def dfs(node):
            visited[node] = True
            tour.append(node)
            for neighbor in mst_adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        dfs(0)
        distance = self.calculate_tour_distance(tour)
        return tour, distance, "2-approximation solution"
    
    def genetic_algorithm(self, population_size=50, generations=100, mutation_rate=0.02):
        """Simple genetic algorithm for TSP"""
        n = len(self.cities)
        
        # Initialize population
        population = []
        for _ in range(population_size):
            tour = list(range(n))
            random.shuffle(tour)
            population.append(tour)
        
        best_tour = None
        best_distance = float('inf')
        
        for generation in range(generations):
            # Calculate fitness (inverse of distance)
            fitness_scores = []
            for tour in population:
                distance = self.calculate_tour_distance(tour)
                fitness = 1 / (1 + distance)
                fitness_scores.append(fitness)
                
                if distance < best_distance:
                    best_distance = distance
                    best_tour = tour[:]
            
            # Selection and crossover
            new_population = []
            
            for _ in range(population_size):
                # Tournament selection
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                # Order crossover
                child = self.order_crossover(parent1, parent2)
                
                # Mutation
                if random.random() < mutation_rate:
                    self.mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        return best_tour, best_distance, f"Genetic algorithm solution (gen: {generations})"
    
    def simulated_annealing(self, initial_temp=10000, cooling_rate=0.995, min_temp=1, max_iterations=10000):
        """Simulated Annealing algorithm for TSP"""
        n = len(self.cities)
        
        # Generate initial random solution
        current_solution = list(range(n))
        random.shuffle(current_solution)
        current_cost = self.calculate_tour_distance(current_solution)
        
        best_solution = current_solution[:]
        best_cost = current_cost
        
        temperature = initial_temp
        iteration = 0
        
        while temperature > min_temp and iteration < max_iterations:
            # Generate neighbor by swapping two random cities
            new_solution = current_solution[:]
            i, j = random.sample(range(n), 2)
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            
            new_cost = self.calculate_tour_distance(new_solution)
            
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
        
        return best_solution, best_cost, f"Simulated annealing solution (temp: {initial_temp}, iterations: {iteration})"
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Tournament selection for genetic algorithm"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_index][:]
    
    def order_crossover(self, parent1, parent2):
        """Order crossover for genetic algorithm"""
        n = len(parent1)
        start, end = sorted(random.sample(range(n), 2))
        
        child = [-1] * n
        child[start:end] = parent1[start:end]
        
        pointer = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                child[pointer % n] = city
                pointer += 1
        
        return child
    
    def mutate(self, tour):
        """Swap mutation for genetic algorithm"""
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]

class TSPVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Solver - NP-Hard Problem Analysis")
        self.root.geometry("1400x900")
        
        self.solver = TSPSolver()
        self.cities = []
        self.current_tour = []
        self.animation_steps = []
        self.current_step = 0
        self.animation_running = False
        
        self.setup_gui()
        self.generate_random_cities(8)
    
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Title
        title_label = ttk.Label(control_frame, text="TSP Solver", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # City generation
        city_frame = ttk.LabelFrame(control_frame, text="Cities", padding=10)
        city_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(city_frame, text="Number of cities:").pack()
        self.city_count = tk.IntVar(value=8)
        city_spinbox = ttk.Spinbox(city_frame, from_=4, to=15, textvariable=self.city_count, width=10)
        city_spinbox.pack(pady=5)
        
        ttk.Button(city_frame, text="Generate Random Cities", 
                  command=self.generate_random_cities_gui).pack(pady=5)
        
        # Algorithm selection
        algo_frame = ttk.LabelFrame(control_frame, text="Algorithms", padding=10)
        algo_frame.pack(fill=tk.X, pady=(0, 10))
        
        algorithms = [
            ("Brute Force (≤10 cities)", self.run_brute_force),
            ("Greedy Nearest Neighbor", self.run_greedy),
            ("MST Approximation", self.run_mst),
            ("Genetic Algorithm", self.run_genetic),
            ("Simulated Annealing", self.run_simulated_annealing)
        ]
        
        for name, command in algorithms:
            ttk.Button(algo_frame, text=name, command=command, width=25).pack(pady=2)
        
        # Animation controls
        anim_frame = ttk.LabelFrame(control_frame, text="Path Simulation", padding=10)
        anim_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(anim_frame, text="🎬 Animate Greedy Algorithm", 
                  command=self.start_greedy_animation, width=25).pack(pady=2)
        
        self.animation_controls_frame = ttk.Frame(anim_frame)
        self.animation_controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(self.animation_controls_frame, text="⏸️ Pause", 
                  command=self.pause_animation, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.animation_controls_frame, text="▶️ Resume", 
                  command=self.resume_animation, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.animation_controls_frame, text="⏹️ Stop", 
                  command=self.stop_animation, width=8).pack(side=tk.LEFT, padx=2)
        
        # Animation speed control
        speed_frame = ttk.Frame(anim_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self.animation_speed = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(speed_frame, from_=0.1, to=3.0, variable=self.animation_speed, 
                               orient=tk.HORIZONTAL, length=150)
        speed_scale.pack(side=tk.LEFT, padx=5)
        ttk.Label(speed_frame, text="3x").pack(side=tk.LEFT)
        
        # Results display
        results_frame = ttk.LabelFrame(control_frame, text="Results", padding=10)
        results_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.result_text = tk.Text(results_frame, height=8, width=30)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Performance analysis
        perf_frame = ttk.LabelFrame(control_frame, text="Performance Analysis", padding=10)
        perf_frame.pack(fill=tk.X)
        
        ttk.Button(perf_frame, text="Run Performance Test", 
                  command=self.run_performance_analysis, width=25).pack(pady=5)
        
        # Right panel for visualization
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Animation status
        self.status_label = ttk.Label(viz_frame, text="Ready for simulation", font=("Arial", 12))
        self.status_label.pack(pady=5)
    
    def generate_random_cities(self, n=None):
        """Generate random cities"""
        if n is None:
            n = self.city_count.get()
        
        self.cities = []
        for i in range(n):
            x = random.uniform(10, 90)
            y = random.uniform(10, 90)
            self.cities.append((x, y))
        
        self.solver.set_cities(self.cities)
        self.current_tour = []
        self.stop_animation()
        self.update_visualization()
    
    def generate_random_cities_gui(self):
        """Generate random cities from GUI"""
        self.generate_random_cities()
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Generated {len(self.cities)} random cities\n")
    
    def update_visualization(self, step_data=None):
        """Update the visualization"""
        self.ax.clear()
        
        if not self.cities:
            return
        
        # Plot cities
        x_coords = [city[0] for city in self.cities]
        y_coords = [city[1] for city in self.cities]
        
        # Color cities based on animation state
        if step_data:
            colors = []
            sizes = []
            for i in range(len(self.cities)):
                if i == step_data.get('current_city'):
                    colors.append('orange')  # Current city
                    sizes.append(200)
                elif i == step_data.get('next_city'):
                    colors.append('lime')    # Next city to visit
                    sizes.append(200)
                elif i in step_data.get('unvisited', set()):
                    colors.append('lightgray')  # Unvisited
                    sizes.append(100)
                else:
                    colors.append('red')     # Visited
                    sizes.append(150)
        else:
            colors = ['red' if i == 0 else 'lightblue' for i in range(len(self.cities))]
            sizes = [150 if i == 0 else 100 for i in range(len(self.cities))]
        
        self.ax.scatter(x_coords, y_coords, c=colors, s=sizes, zorder=3, edgecolors='black', linewidth=1)
        
        # Label cities
        for i, (x, y) in enumerate(self.cities):
            self.ax.annotate(str(i), (x, y), xytext=(0, 0), textcoords='offset points', 
                           ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Plot tour if available
        if step_data and 'tour' in step_data and len(step_data['tour']) > 1:
            tour = step_data['tour']
            # Plot completed path
            for i in range(len(tour) - 1):
                city1 = self.cities[tour[i]]
                city2 = self.cities[tour[i + 1]]
                self.ax.plot([city1[0], city2[0]], [city1[1], city2[1]], 'b-', linewidth=3, alpha=0.7)
            
            # Highlight current edge being considered
            if 'next_city' in step_data:
                current = self.cities[step_data['current_city']]
                next_city = self.cities[step_data['next_city']]
                self.ax.plot([current[0], next_city[0]], [current[1], next_city[1]], 
                           'r--', linewidth=4, alpha=0.8, label='Current edge')
        
        elif self.current_tour:
            # Static tour display
            tour_x = [self.cities[i][0] for i in self.current_tour] + [self.cities[self.current_tour[0]][0]]
            tour_y = [self.cities[i][1] for i in self.current_tour] + [self.cities[self.current_tour[0]][1]]
            self.ax.plot(tour_x, tour_y, 'b-', linewidth=2, alpha=0.7)
        
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_title("Traveling Salesman Problem - Path Simulation", fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # Add legend for animation
        if step_data:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Current City'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label='Next City'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Visited'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Unvisited'),
                plt.Line2D([0], [0], color='blue', linewidth=3, label='Completed Path'),
                plt.Line2D([0], [0], color='red', linestyle='--', linewidth=3, label='Current Edge')
            ]
            self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        self.canvas.draw()
    
    def start_greedy_animation(self):
        """Start animated greedy algorithm"""
        if not self.cities:
            messagebox.showwarning("Warning", "Please generate cities first!")
            return
        
        if self.animation_running:
            self.stop_animation()
        
        # Get animation steps
        _, _, _, self.animation_steps = self.solver.greedy_nearest_neighbor_animated()
        self.current_step = 0
        self.animation_running = True
        
        self.result_text.insert(tk.END, "\n🎬 Starting Greedy Algorithm Animation...\n")
        self.result_text.insert(tk.END, f"Total steps: {len(self.animation_steps)}\n")
        self.result_text.see(tk.END)
        
        self.animate_step()
    
    def animate_step(self):
        """Animate one step of the algorithm"""
        if not self.animation_running or self.current_step >= len(self.animation_steps):
            if self.current_step >= len(self.animation_steps):
                self.animation_complete()
            return
        
        step_data = self.animation_steps[self.current_step]
        self.update_visualization(step_data)
        
        # Update status
        message = step_data.get('message', f"Step {self.current_step + 1}")
        self.status_label.config(text=f"Step {self.current_step + 1}/{len(self.animation_steps)}: {message}")
        
        # Show distance if available
        if 'distance' in step_data:
            self.result_text.insert(tk.END, f"Step {self.current_step + 1}: {message}\n")
            self.result_text.see(tk.END)
        
        self.current_step += 1
        
        # Schedule next step
        delay = int(1000 / self.animation_speed.get())  # Convert speed to delay
        self.root.after(delay, self.animate_step)
    
    def animation_complete(self):
        """Handle animation completion"""
        self.animation_running = False
        total_distance = self.solver.calculate_tour_distance(self.animation_steps[-1]['tour'][:-1])  # Remove duplicate start city
        
        self.status_label.config(text=f"✅ Animation Complete! Total Distance: {total_distance:.2f}")
        self.result_text.insert(tk.END, f"\n✅ Greedy Algorithm Complete!\n")
        self.result_text.insert(tk.END, f"Final tour distance: {total_distance:.2f}\n")
        self.result_text.insert(tk.END, "-" * 40 + "\n")
        self.result_text.see(tk.END)
        
        # Store final tour for static display
        self.current_tour = self.animation_steps[-1]['tour'][:-1]  # Remove duplicate start city
    
    def pause_animation(self):
        """Pause the animation"""
        self.animation_running = False
        self.status_label.config(text="⏸️ Animation Paused")
    
    def resume_animation(self):
        """Resume the animation"""
        if self.animation_steps and self.current_step < len(self.animation_steps):
            self.animation_running = True
            self.status_label.config(text="▶️ Animation Resumed")
            self.animate_step()
    
    def stop_animation(self):
        """Stop the animation"""
        self.animation_running = False
        self.animation_steps = []
        self.current_step = 0
        self.status_label.config(text="⏹️ Animation Stopped")
        self.update_visualization()
    
    def run_algorithm(self, algorithm_func, algorithm_name):
        """Run an algorithm and display results"""
        if not self.cities:
            messagebox.showwarning("Warning", "Please generate cities first!")
            return
        
        self.stop_animation()  # Stop any running animation
        
        self.result_text.insert(tk.END, f"\nRunning {algorithm_name}...\n")
        self.root.update()
        
        start_time = time.time()
        try:
            tour, distance, message = algorithm_func()
            end_time = time.time()
            
            if tour is None:
                self.result_text.insert(tk.END, f"Error: {message}\n")
                return
            
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            self.current_tour = tour
            self.update_visualization()
            
            # Display results
            self.result_text.insert(tk.END, f"Algorithm: {algorithm_name}\n")
            self.result_text.insert(tk.END, f"Tour: {' -> '.join(map(str, tour))} -> {tour[0]}\n")
            self.result_text.insert(tk.END, f"Distance: {distance:.2f}\n")
            self.result_text.insert(tk.END, f"Time: {execution_time:.2f} ms\n")
            self.result_text.insert(tk.END, f"Status: {message}\n")
            self.result_text.insert(tk.END, "-" * 30 + "\n")
            
            # Auto-scroll to bottom
            self.result_text.see(tk.END)
            
        except Exception as e:
            self.result_text.insert(tk.END, f"Error: {str(e)}\n")
    
    def run_brute_force(self):
        """Run brute force algorithm"""
        self.run_algorithm(self.solver.brute_force_tsp, "Brute Force")
    
    def run_greedy(self):
        """Run greedy nearest neighbor algorithm"""
        self.run_algorithm(self.solver.greedy_nearest_neighbor, "Greedy Nearest Neighbor")
    
    def run_mst(self):
        """Run MST approximation algorithm"""
        self.run_algorithm(self.solver.mst_approximation, "MST Approximation")
    
    def run_genetic(self):
        """Run genetic algorithm"""
        self.run_algorithm(lambda: self.solver.genetic_algorithm(generations=50), "Genetic Algorithm")
    
    def run_simulated_annealing(self):
        """Run simulated annealing algorithm"""
        self.run_algorithm(lambda: self.solver.simulated_annealing(initial_temp=1000, max_iterations=5000), "Simulated Annealing")
    
    def run_performance_analysis(self):
        """Run performance analysis and show results"""
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Performance Analysis")
        analysis_window.geometry("800x600")
        
        # Create matplotlib figure for analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        canvas = FigureCanvasTkAgg(fig, analysis_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Run tests
        city_sizes = [4, 5, 6, 7, 8, 10, 12, 15]
        algorithms = {
            'Greedy': self.solver.greedy_nearest_neighbor,
            'MST': self.solver.mst_approximation,
            'Simulated Annealing': lambda: self.solver.simulated_annealing(initial_temp=1000, max_iterations=1000)
        }
        
        results = {name: {'sizes': [], 'times': [], 'distances': []} for name in algorithms}
        
        # Add brute force for small sizes
        brute_force_sizes = []
        brute_force_times = []
        brute_force_distances = []
        
        for size in city_sizes:
            # Generate test cities
            test_cities = [(random.uniform(10, 90), random.uniform(10, 90)) for _ in range(size)]
            self.solver.set_cities(test_cities)
            
            # Test brute force for small sizes
            if size <= 8:
                start_time = time.time()
                tour, distance, _ = self.solver.brute_force_tsp()
                end_time = time.time()
                
                if tour is not None:
                    brute_force_sizes.append(size)
                    brute_force_times.append((end_time - start_time) * 1000)
                    brute_force_distances.append(distance)
            
            # Test other algorithms
            for name, func in algorithms.items():
                start_time = time.time()
                tour, distance, _ = func()
                end_time = time.time()
                
                results[name]['sizes'].append(size)
                results[name]['times'].append((end_time - start_time) * 1000)
                results[name]['distances'].append(distance)
        
        # Plot execution times
        ax1.plot(brute_force_sizes, brute_force_times, 'r-o', label='Brute Force', linewidth=2)
        for name, data in results.items():
            ax1.plot(data['sizes'], data['times'], '-o', label=name, linewidth=2)
        ax1.set_xlabel('Number of Cities')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Time Complexity Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot solution quality
        ax2.plot(brute_force_sizes, brute_force_distances, 'r-o', label='Brute Force (Optimal)', linewidth=2)
        for name, data in results.items():
            ax2.plot(data['sizes'], data['distances'], '-o', label=name, linewidth=2)
        ax2.set_xlabel('Number of Cities')
        ax2.set_ylabel('Tour Distance')
        ax2.set_title('Solution Quality Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Complexity growth chart
        theoretical_sizes = list(range(4, 11))
        factorial_times = [math.factorial(n-1) / 1000000 for n in theoretical_sizes]  # Normalized
        quadratic_times = [n*n / 10 for n in theoretical_sizes]  # Normalized
        
        ax3.plot(theoretical_sizes, factorial_times, 'r--', label='O(n!) - Brute Force', linewidth=2)
        ax3.plot(theoretical_sizes, quadratic_times, 'g--', label='O(n²) - Greedy', linewidth=2)
        ax3.set_xlabel('Number of Cities')
        ax3.set_ylabel('Relative Time Complexity')
        ax3.set_title('Theoretical Complexity Growth')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Algorithm comparison table
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = [
            ['Algorithm', 'Time Complexity', 'Space Complexity', 'Solution Quality'],
            ['Brute Force', 'O(n!)', 'O(n)', 'Optimal'],
            ['Greedy NN', 'O(n²)', 'O(n)', 'Heuristic'],
            ['MST Approx', 'O(n² log n)', 'O(n²)', '≤ 2 × Optimal'],
            ['Genetic Alg', 'O(g × p × n)', 'O(p × n)', 'Heuristic'],
            ['Simulated Annealing', 'O(k × n)', 'O(n)', 'Heuristic']
        ]
        
        table = ax4.table(cellText=table_data[1:], colLabels=table_data[0], 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Algorithm Comparison Summary')
        
        plt.tight_layout()
        canvas.draw()

def main():
    """Main function to run the TSP application"""
    root = tk.Tk()
    app = TSPVisualizer(root)
    
    # Add some information about the project
    info_text = """TSP Solver - NP-Hard Problem Analysis
=====================================

This project demonstrates different approaches to solving
the Traveling Salesman Problem (TSP), a classic NP-Hard problem.

Algorithms implemented:
1. Brute Force - O(n!) - Optimal but slow
2. Greedy Nearest Neighbor - O(n²) - Fast heuristic
3. MST Approximation - O(n² log n) - 2-approximation
4. Genetic Algorithm - Metaheuristic approach
5. Simulated Annealing - Metaheuristic approach

🎬 NEW: Path Simulation Feature!
- Click "Animate Greedy Algorithm" to see step-by-step path building
- Control animation speed and playback
- Visual legend shows current city, next city, and path progress

Instructions:
1. Generate random cities or adjust the number
2. Run different algorithms to compare results
3. Use "Animate Greedy Algorithm" for step-by-step visualization
4. Use Performance Analysis to see complexity differences

Note: Brute force is limited to ≤10 cities due to factorial complexity.
"""
    
    app.result_text.insert(tk.END, info_text)
    
    root.mainloop()

if __name__ == "__main__":
    main()
