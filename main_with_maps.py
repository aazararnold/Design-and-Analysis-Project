"""
Traveling Salesman Problem (TSP) with Google Maps Integration
Design and Analysis of Algorithms Course Project

This enhanced version uses Google Maps API for:
- Real locations and addresses
- Actual driving distances and times
- Interactive map visualization
- Route optimization with real-world data
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import numpy as np
import random
import itertools
import time
import math
import requests
import json
import webbrowser
import urllib.parse
from typing import List, Tuple, Dict, Optional

class GoogleMapsAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api"
        
    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Convert address to latitude/longitude coordinates"""
        url = f"{self.base_url}/geocode/json"
        params = {
            'address': address,
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] == 'OK' and data['results']:
                location = data['results'][0]['geometry']['location']
                return (location['lat'], location['lng'])
            else:
                print(f"Geocoding failed for {address}: {data.get('status', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"Error geocoding {address}: {e}")
            return None
    
    def get_distance_matrix(self, origins: List[Tuple[float, float]], 
                          destinations: List[Tuple[float, float]], 
                          mode: str = 'driving') -> Optional[Dict]:
        """Get distance and duration matrix between locations"""
        url = f"{self.base_url}/distancematrix/json"
        
        # Convert coordinates to string format
        origins_str = '|'.join([f"{lat},{lng}" for lat, lng in origins])
        destinations_str = '|'.join([f"{lat},{lng}" for lat, lng in destinations])
        
        params = {
            'origins': origins_str,
            'destinations': destinations_str,
            'mode': mode,
            'units': 'metric',
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] == 'OK':
                return data
            else:
                print(f"Distance Matrix API failed: {data.get('status', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"Error getting distance matrix: {e}")
            return None
    
    def generate_maps_url(self, waypoints: List[Tuple[float, float]], 
                         addresses: List[str] = None) -> str:
        """Generate Google Maps URL for the route"""
        if not waypoints:
            return ""
        
        # Start with the first waypoint
        origin = f"{waypoints[0][0]},{waypoints[0][1]}"
        destination = f"{waypoints[-1][0]},{waypoints[-1][1]}"
        
        # Add intermediate waypoints
        if len(waypoints) > 2:
            waypoints_str = '|'.join([f"{lat},{lng}" for lat, lng in waypoints[1:-1]])
            url = f"https://www.google.com/maps/dir/{origin}/{waypoints_str}/{destination}"
        else:
            url = f"https://www.google.com/maps/dir/{origin}/{destination}"
        
        return url

class RealWorldTSPSolver:
    def __init__(self, google_maps_api: GoogleMapsAPI):
        self.google_maps = google_maps_api
        self.locations = []  # List of (name, lat, lng)
        self.distance_matrix = []  # Distance matrix in meters
        self.duration_matrix = []  # Duration matrix in seconds
        self.use_duration = False  # Whether to optimize for time vs distance
        
    def add_location(self, name: str, address: str = None, 
                    lat: float = None, lng: float = None) -> bool:
        """Add a location by address or coordinates"""
        if address:
            coords = self.google_maps.geocode_address(address)
            if coords:
                self.locations.append((name, coords[0], coords[1], address))
                return True
            else:
                return False
        elif lat is not None and lng is not None:
            self.locations.append((name, lat, lng, f"{lat},{lng}"))
            return True
        else:
            return False
    
    def clear_locations(self):
        """Clear all locations"""
        self.locations = []
        self.distance_matrix = []
        self.duration_matrix = []
    
    def build_matrices(self, mode: str = 'driving') -> bool:
        """Build distance and duration matrices using Google Maps API"""
        if len(self.locations) < 2:
            return False
        
        coords = [(loc[1], loc[2]) for loc in self.locations]
        matrix_data = self.google_maps.get_distance_matrix(coords, coords, mode)
        
        if not matrix_data:
            return False
        
        n = len(self.locations)
        self.distance_matrix = [[0] * n for _ in range(n)]
        self.duration_matrix = [[0] * n for _ in range(n)]
        
        for i, row in enumerate(matrix_data['rows']):
            for j, element in enumerate(row['elements']):
                if element['status'] == 'OK':
                    # Distance in meters
                    self.distance_matrix[i][j] = element['distance']['value']
                    # Duration in seconds
                    self.duration_matrix[i][j] = element['duration']['value']
                else:
                    # Use large value for unreachable destinations
                    self.distance_matrix[i][j] = float('inf')
                    self.duration_matrix[i][j] = float('inf')
        
        return True
    
    def get_matrix(self):
        """Get the appropriate matrix based on optimization preference"""
        return self.duration_matrix if self.use_duration else self.distance_matrix
    
    def calculate_tour_cost(self, tour: List[int]) -> float:
        """Calculate total cost (distance or time) of a tour"""
        matrix = self.get_matrix()
        total_cost = 0
        
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            total_cost += matrix[current_city][next_city]
        
        return total_cost
    
    def format_cost(self, cost: float) -> str:
        """Format cost for display"""
        if self.use_duration:
            hours = int(cost // 3600)
            minutes = int((cost % 3600) // 60)
            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
        else:
            if cost >= 1000:
                return f"{cost/1000:.1f} km"
            else:
                return f"{cost:.0f} m"
    
    def greedy_nearest_neighbor_animated(self, start_city: int = 0):
        """Greedy algorithm with animation data for real locations"""
        n = len(self.locations)
        if n < 2:
            return None, float('inf'), "Need at least 2 locations", []
        
        matrix = self.get_matrix()
        unvisited = set(range(n))
        tour = [start_city]
        unvisited.remove(start_city)
        current_city = start_city
        
        animation_steps = []
        animation_steps.append({
            'current_city': current_city,
            'tour': tour[:],
            'unvisited': unvisited.copy(),
            'message': f"Starting at {self.locations[start_city][0]}"
        })
        
        while unvisited:
            # Find nearest unvisited location
            nearest_city = min(unvisited, key=lambda city: matrix[current_city][city])
            nearest_cost = matrix[current_city][nearest_city]
            
            tour.append(nearest_city)
            unvisited.remove(nearest_city)
            
            animation_steps.append({
                'current_city': current_city,
                'next_city': nearest_city,
                'tour': tour[:],
                'unvisited': unvisited.copy(),
                'cost': nearest_cost,
                'message': f"From {self.locations[current_city][0]} → {self.locations[nearest_city][0]} ({self.format_cost(nearest_cost)})"
            })
            
            current_city = nearest_city
        
        # Return to start
        final_cost = matrix[current_city][start_city]
        animation_steps.append({
            'current_city': current_city,
            'next_city': start_city,
            'tour': tour + [start_city],
            'unvisited': set(),
            'cost': final_cost,
            'message': f"Return to start: {self.locations[current_city][0]} → {self.locations[start_city][0]} ({self.format_cost(final_cost)})"
        })
        
        total_cost = self.calculate_tour_cost(tour)
        return tour, total_cost, "Real-world heuristic solution", animation_steps
    
    def brute_force_tsp(self):
        """Brute force for small number of locations"""
        n = len(self.locations)
        if n > 8:
            return None, float('inf'), "Too many locations for brute force (max 8)"
        
        matrix = self.get_matrix()
        min_cost = float('inf')
        best_tour = None
        
        for perm in itertools.permutations(range(1, n)):
            tour = [0] + list(perm)
            cost = self.calculate_tour_cost(tour)
            
            if cost < min_cost:
                min_cost = cost
                best_tour = tour
        
        return best_tour, min_cost, "Optimal real-world solution"

class RealWorldTSPVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-World TSP Solver with Google Maps")
        self.root.geometry("1600x1000")
        
        # Initialize with empty API key
        self.google_maps = None
        self.solver = None
        self.animation_steps = []
        self.current_step = 0
        self.animation_running = False
        
        self.setup_gui()
        self.setup_sample_locations()
    
    def setup_gui(self):
        """Setup the enhanced GUI with Maps integration"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Title
        title_label = ttk.Label(control_frame, text="Real-World TSP Solver", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # API Key setup
        api_frame = ttk.LabelFrame(control_frame, text="Google Maps API Setup", padding=10)
        api_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(api_frame, text="API Key:").pack(anchor=tk.W)
        self.api_key_var = tk.StringVar()
        api_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, show="*", width=30)
        api_entry.pack(fill=tk.X, pady=2)
        
        ttk.Button(api_frame, text="Set API Key", command=self.set_api_key).pack(pady=2)
        ttk.Button(api_frame, text="Get API Key", command=self.open_api_help).pack(pady=2)
        
        self.api_status = ttk.Label(api_frame, text="❌ API Key not set", foreground="red")
        self.api_status.pack(pady=2)
        
        # Location management
        location_frame = ttk.LabelFrame(control_frame, text="Locations", padding=10)
        location_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add location controls
        add_frame = ttk.Frame(location_frame)
        add_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(add_frame, text="Name:").pack(side=tk.LEFT)
        self.location_name = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.location_name, width=15).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(add_frame, text="Address:").pack(side=tk.LEFT)
        self.location_address = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.location_address, width=20).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(location_frame, text="Add Location", command=self.add_location).pack(pady=2)
        ttk.Button(location_frame, text="Load Sample Locations", command=self.load_sample_locations).pack(pady=2)
        ttk.Button(location_frame, text="Clear All", command=self.clear_locations).pack(pady=2)
        
        # Location list
        self.location_listbox = tk.Listbox(location_frame, height=6)
        self.location_listbox.pack(fill=tk.X, pady=2)
        
        # Optimization settings
        opt_frame = ttk.LabelFrame(control_frame, text="Optimization", padding=10)
        opt_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.optimize_for = tk.StringVar(value="distance")
        ttk.Radiobutton(opt_frame, text="Minimize Distance", variable=self.optimize_for, 
                       value="distance").pack(anchor=tk.W)
        ttk.Radiobutton(opt_frame, text="Minimize Time", variable=self.optimize_for, 
                       value="time").pack(anchor=tk.W)
        
        self.travel_mode = tk.StringVar(value="driving")
        ttk.Label(opt_frame, text="Travel Mode:").pack(anchor=tk.W, pady=(5,0))
        mode_frame = ttk.Frame(opt_frame)
        mode_frame.pack(fill=tk.X)
        ttk.Radiobutton(mode_frame, text="Driving", variable=self.travel_mode, 
                       value="driving").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Walking", variable=self.travel_mode, 
                       value="walking").pack(side=tk.LEFT)
        
        # Algorithm controls
        algo_frame = ttk.LabelFrame(control_frame, text="Algorithms", padding=10)
        algo_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(algo_frame, text="🔄 Build Distance Matrix", 
                  command=self.build_matrices, width=25).pack(pady=2)
        ttk.Button(algo_frame, text="🎬 Animate Greedy Algorithm", 
                  command=self.start_animation, width=25).pack(pady=2)
        ttk.Button(algo_frame, text="⚡ Run Brute Force", 
                  command=self.run_brute_force, width=25).pack(pady=2)
        
        # Animation controls
        anim_control_frame = ttk.Frame(algo_frame)
        anim_control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(anim_control_frame, text="⏸️", command=self.pause_animation, width=4).pack(side=tk.LEFT, padx=1)
        ttk.Button(anim_control_frame, text="▶️", command=self.resume_animation, width=4).pack(side=tk.LEFT, padx=1)
        ttk.Button(anim_control_frame, text="⏹️", command=self.stop_animation, width=4).pack(side=tk.LEFT, padx=1)
        
        # Speed control
        ttk.Label(algo_frame, text="Animation Speed:").pack(anchor=tk.W)
        self.animation_speed = tk.DoubleVar(value=1.0)
        ttk.Scale(algo_frame, from_=0.1, to=3.0, variable=self.animation_speed, 
                 orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Maps integration
        maps_frame = ttk.LabelFrame(control_frame, text="Google Maps", padding=10)
        maps_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(maps_frame, text="🗺️ Open Route in Google Maps", 
                  command=self.open_in_google_maps, width=25).pack(pady=2)
        ttk.Button(maps_frame, text="📍 View All Locations", 
                  command=self.view_all_locations, width=25).pack(pady=2)
        
        # Results display
        results_frame = ttk.LabelFrame(control_frame, text="Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(results_frame, height=12, width=35)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel for visualization
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_label = ttk.Label(viz_frame, text="Ready - Set up Google Maps API key to begin", 
                                     font=("Arial", 12))
        self.status_label.pack(pady=5)
    
    def set_api_key(self):
        """Set up Google Maps API"""
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showwarning("Warning", "Please enter a valid API key")
            return
        
        try:
            self.google_maps = GoogleMapsAPI(api_key)
            self.solver = RealWorldTSPSolver(self.google_maps)
            self.api_status.config(text="✅ API Key set successfully", foreground="green")
            self.status_label.config(text="Ready - Add locations to begin")
            
            self.result_text.insert(tk.END, "✅ Google Maps API initialized successfully!\n")
            self.result_text.insert(tk.END, "You can now add real locations and addresses.\n\n")
            
        except Exception as e:
            self.api_status.config(text="❌ API Key error", foreground="red")
            messagebox.showerror("Error", f"Failed to initialize Google Maps API: {e}")
    
    def open_api_help(self):
        """Open Google Maps API documentation"""
        webbrowser.open("https://developers.google.com/maps/documentation/distance-matrix/get-api-key")
    
    def setup_sample_locations(self):
        """Setup sample locations for demo"""
        self.sample_locations = [
            ("New York, NY", "Times Square, New York, NY"),
            ("Philadelphia, PA", "Liberty Bell, Philadelphia, PA"),
            ("Washington, DC", "White House, Washington, DC"),
            ("Baltimore, MD", "Inner Harbor, Baltimore, MD"),
            ("Boston, MA", "Fenway Park, Boston, MA"),
            ("San Francisco, CA", "Golden Gate Bridge, San Francisco, CA"),
            ("Los Angeles, CA", "Hollywood Sign, Los Angeles, CA"),
            ("Las Vegas, NV", "Las Vegas Strip, Las Vegas, NV"),
            ("Chicago, IL", "Millennium Park, Chicago, IL"),
            ("Miami, FL", "South Beach, Miami, FL")
        ]
    
    def add_location(self):
        """Add a new location"""
        if not self.solver:
            messagebox.showwarning("Warning", "Please set up Google Maps API first")
            return
        
        name = self.location_name.get().strip()
        address = self.location_address.get().strip()
        
        if not name or not address:
            messagebox.showwarning("Warning", "Please enter both name and address")
            return
        
        self.result_text.insert(tk.END, f"Adding location: {name} at {address}...\n")
        self.root.update()
        
        if self.solver.add_location(name, address=address):
            self.location_listbox.insert(tk.END, f"{name} - {address}")
            self.location_name.set("")
            self.location_address.set("")
            self.result_text.insert(tk.END, f"✅ Added: {name}\n")
            self.update_visualization()
        else:
            self.result_text.insert(tk.END, f"❌ Failed to add: {name}\n")
        
        self.result_text.see(tk.END)
    
    def load_sample_locations(self):
        """Load sample locations for demo"""
        if not self.solver:
            messagebox.showwarning("Warning", "Please set up Google Maps API first")
            return
        
        # Show selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Sample Locations")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="Select locations to add:").pack(pady=10)
        
        # Checkboxes for each sample location
        vars_list = []
        for name, address in self.sample_locations:
            var = tk.BooleanVar()
            ttk.Checkbutton(dialog, text=f"{name} - {address}", variable=var).pack(anchor=tk.W, padx=20)
            vars_list.append((var, name, address))
        
        def add_selected():
            selected = [(name, addr) for var, name, addr in vars_list if var.get()]
            dialog.destroy()
            
            if selected:
                self.result_text.insert(tk.END, f"Adding {len(selected)} sample locations...\n")
                self.root.update()
                
                for name, address in selected:
                    if self.solver.add_location(name, address=address):
                        self.location_listbox.insert(tk.END, f"{name} - {address}")
                        self.result_text.insert(tk.END, f"✅ Added: {name}\n")
                    else:
                        self.result_text.insert(tk.END, f"❌ Failed: {name}\n")
                    self.root.update()
                
                self.update_visualization()
                self.result_text.see(tk.END)
        
        ttk.Button(dialog, text="Add Selected", command=add_selected).pack(pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack()
    
    def clear_locations(self):
        """Clear all locations"""
        if self.solver:
            self.solver.clear_locations()
        self.location_listbox.delete(0, tk.END)
        self.stop_animation()
        self.update_visualization()
        self.result_text.insert(tk.END, "🗑️ Cleared all locations\n")
    
    def build_matrices(self):
        """Build distance and duration matrices"""
        if not self.solver or len(self.solver.locations) < 2:
            messagebox.showwarning("Warning", "Please add at least 2 locations first")
            return
        
        self.result_text.insert(tk.END, f"\n🔄 Building distance matrix for {len(self.solver.locations)} locations...\n")
        self.result_text.insert(tk.END, f"Travel mode: {self.travel_mode.get()}\n")
        self.root.update()
        
        # Set optimization preference
        self.solver.use_duration = (self.optimize_for.get() == "time")
        
        if self.solver.build_matrices(self.travel_mode.get()):
            self.result_text.insert(tk.END, "✅ Distance matrix built successfully!\n")
            self.result_text.insert(tk.END, f"Optimizing for: {'travel time' if self.solver.use_duration else 'distance'}\n")
            self.status_label.config(text="Ready - Distance matrix built, you can now run algorithms")
        else:
            self.result_text.insert(tk.END, "❌ Failed to build distance matrix\n")
        
        self.result_text.see(tk.END)
    
    def update_visualization(self):
        """Update the map visualization"""
        self.ax.clear()
        
        if not self.solver or not self.solver.locations:
            self.ax.text(0.5, 0.5, 'Add locations to see map visualization', 
                        ha='center', va='center', transform=self.ax.transAxes, fontsize=14)
            self.canvas.draw()
            return
        
        # Plot locations
        lats = [loc[1] for loc in self.solver.locations]
        lngs = [loc[2] for loc in self.solver.locations]
        names = [loc[0] for loc in self.solver.locations]
        
        # Color coding for animation
        colors = ['red' if i == 0 else 'lightblue' for i in range(len(self.solver.locations))]
        sizes = [200 if i == 0 else 100 for i in range(len(self.solver.locations))]
        
        self.ax.scatter(lngs, lats, c=colors, s=sizes, zorder=3, edgecolors='black', linewidth=1)
        
        # Label locations
        for i, (name, lat, lng, _) in enumerate(self.solver.locations):
            self.ax.annotate(f"{i}: {name}", (lng, lat), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, fontweight='bold')
        
        # Plot tour if available
        if hasattr(self, 'current_tour') and self.current_tour:
            tour_lngs = [self.solver.locations[i][2] for i in self.current_tour] + [self.solver.locations[self.current_tour[0]][2]]
            tour_lats = [self.solver.locations[i][1] for i in self.current_tour] + [self.solver.locations[self.current_tour[0]][1]]
            self.ax.plot(tour_lngs, tour_lats, 'b-', linewidth=2, alpha=0.7, label='TSP Route')
        
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_title('Real-World TSP - Location Map')
        self.ax.grid(True, alpha=0.3)
        
        # Auto-adjust view
        if lats and lngs:
            lat_margin = (max(lats) - min(lats)) * 0.1
            lng_margin = (max(lngs) - min(lngs)) * 0.1
            self.ax.set_xlim(min(lngs) - lng_margin, max(lngs) + lng_margin)
            self.ax.set_ylim(min(lats) - lat_margin, max(lats) + lat_margin)
        
        self.canvas.draw()
    
    def start_animation(self):
        """Start animated greedy algorithm"""
        if not self.solver or not self.solver.locations:
            messagebox.showwarning("Warning", "Please add locations first")
            return
        
        if not self.solver.distance_matrix:
            messagebox.showwarning("Warning", "Please build distance matrix first")
            return
        
        if self.animation_running:
            self.stop_animation()
        
        self.result_text.insert(tk.END, "\n🎬 Starting Real-World Greedy Algorithm Animation...\n")
        self.root.update()
        
        _, _, _, self.animation_steps = self.solver.greedy_nearest_neighbor_animated()
        self.current_step = 0
        self.animation_running = True
        
        self.animate_step()
    
    def animate_step(self):
        """Animate one step"""
        if not self.animation_running or self.current_step >= len(self.animation_steps):
            if self.current_step >= len(self.animation_steps):
                self.animation_complete()
            return
        
        step_data = self.animation_steps[self.current_step]
        
        # Update status
        message = step_data.get('message', f"Step {self.current_step + 1}")
        self.status_label.config(text=f"Step {self.current_step + 1}/{len(self.animation_steps)}: {message}")
        
        if 'cost' in step_data:
            self.result_text.insert(tk.END, f"Step {self.current_step + 1}: {message}\n")
            self.result_text.see(tk.END)
        
        self.current_step += 1
        
        # Schedule next step
        delay = int(2000 / self.animation_speed.get())  # Slower for real-world data
        self.root.after(delay, self.animate_step)
    
    def animation_complete(self):
        """Handle animation completion"""
        self.animation_running = False
        if self.animation_steps:
            final_tour = self.animation_steps[-1]['tour'][:-1]  # Remove duplicate
            total_cost = self.solver.calculate_tour_cost(final_tour)
            
            self.current_tour = final_tour
            self.update_visualization()
            
            cost_str = self.solver.format_cost(total_cost)
            self.status_label.config(text=f"✅ Animation Complete! Total: {cost_str}")
            self.result_text.insert(tk.END, f"\n✅ Real-World Greedy Algorithm Complete!\n")
            self.result_text.insert(tk.END, f"Final tour cost: {cost_str}\n")
            self.result_text.insert(tk.END, "-" * 40 + "\n")
            self.result_text.see(tk.END)
    
    def pause_animation(self):
        """Pause animation"""
        self.animation_running = False
        self.status_label.config(text="⏸️ Animation Paused")
    
    def resume_animation(self):
        """Resume animation"""
        if self.animation_steps and self.current_step < len(self.animation_steps):
            self.animation_running = True
            self.status_label.config(text="▶️ Animation Resumed")
            self.animate_step()
    
    def stop_animation(self):
        """Stop animation"""
        self.animation_running = False
        self.animation_steps = []
        self.current_step = 0
        self.status_label.config(text="⏹️ Animation Stopped")
    
    def run_brute_force(self):
        """Run brute force algorithm"""
        if not self.solver or not self.solver.locations:
            messagebox.showwarning("Warning", "Please add locations first")
            return
        
        if not self.solver.distance_matrix:
            messagebox.showwarning("Warning", "Please build distance matrix first")
            return
        
        self.result_text.insert(tk.END, "\n⚡ Running Brute Force Algorithm...\n")
        self.root.update()
        
        start_time = time.time()
        tour, cost, message = self.solver.brute_force_tsp()
        end_time = time.time()
        
        if tour:
            self.current_tour = tour
            self.update_visualization()
            
            cost_str = self.solver.format_cost(cost)
            execution_time = (end_time - start_time) * 1000
            
            self.result_text.insert(tk.END, f"✅ Brute Force Complete!\n")
            self.result_text.insert(tk.END, f"Optimal tour cost: {cost_str}\n")
            self.result_text.insert(tk.END, f"Execution time: {execution_time:.2f} ms\n")
            self.result_text.insert(tk.END, f"Tour: {' → '.join([self.solver.locations[i][0] for i in tour])}\n")
        else:
            self.result_text.insert(tk.END, f"❌ {message}\n")
        
        self.result_text.see(tk.END)
    
    def open_in_google_maps(self):
        """Open current route in Google Maps"""
        if not hasattr(self, 'current_tour') or not self.current_tour:
            messagebox.showwarning("Warning", "No route to display. Run an algorithm first.")
            return
        
        # Get coordinates for the tour
        waypoints = [(self.solver.locations[i][1], self.solver.locations[i][2]) for i in self.current_tour]
        waypoints.append(waypoints[0])  # Return to start
        
        url = self.google_maps.generate_maps_url(waypoints)
        if url:
            webbrowser.open(url)
            self.result_text.insert(tk.END, "🗺️ Opened route in Google Maps\n")
        else:
            messagebox.showerror("Error", "Failed to generate Google Maps URL")
    
    def view_all_locations(self):
        """View all locations in Google Maps"""
        if not self.solver or not self.solver.locations:
            messagebox.showwarning("Warning", "No locations to display")
            return
        
        # Create URL with all locations as waypoints
        coords = [(loc[1], loc[2]) for loc in self.solver.locations]
        url = self.google_maps.generate_maps_url(coords)
        
        if url:
            webbrowser.open(url)
            self.result_text.insert(tk.END, "📍 Opened all locations in Google Maps\n")

def main():
    """Main function"""
    root = tk.Tk()
    app = RealWorldTSPVisualizer(root)
    
    # Add welcome message
    welcome_text = """🌍 Real-World TSP Solver with Google Maps Integration
================================================

This enhanced version uses Google Maps API for real-world optimization!

🔧 Setup Instructions:
1. Get a Google Maps API key (click "Get API Key" button)
2. Enable Distance Matrix API and Geocoding API
3. Enter your API key and click "Set API Key"
4. Add real locations using addresses
5. Build distance matrix for your travel mode
6. Run algorithms to find optimal routes!

🎯 Features:
• Real addresses and locations
• Actual driving distances and times
• Multiple travel modes (driving, walking)
• Optimize for distance or time
• Animated step-by-step visualization
• Open results directly in Google Maps
• Sample locations for quick testing

📍 Sample locations available for major US cities!

Note: You need a valid Google Maps API key to use real-world features.
For testing without API, use the original version (main.py).
"""
    
    app.result_text.insert(tk.END, welcome_text)
    
    root.mainloop()

if __name__ == "__main__":
    main()
