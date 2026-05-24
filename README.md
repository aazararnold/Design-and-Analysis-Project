# Traveling Salesman Problem (TSP) Solver & Performance Analyzer 🗺️📊

An interactive graphical visualizer and benchmarking suite built to implement, analyze, and compare multiple approaches to solving the classic **Traveling Salesman Problem (TSP)**—a cornerstone problem in computer science classified as NP-Hard.

Developed as a comprehensive project for **Design and Analysis of Algorithms**, this suite helps visualize why exact brute-force solutions fail at scale and how heuristics, approximations, and metaheuristics offer practical alternatives.

---

## 🚀 Implemented Algorithms

| Algorithm | Type | Time Complexity | Space Complexity | Solution Quality |
| :--- | :--- | :--- | :--- | :--- |
| **Brute Force** | Exact / Deterministic | $O(n!)$ | $O(n)$ | Strictly Optimal |
| **Greedy Nearest Neighbor** | Heuristic | $O(n^2)$ | $O(n)$ | Variable / Empirical |
| **MST Approximation** | 2-Approximation | $O(n^2 \log n)$ | $O(n^2)$ | Guaranteed $\le 2 	imes$ Optimal |
| **Genetic Algorithm** | Metaheuristic | $O(g 	imes p 	imes n)$ | $O(p 	imes n)$ | Near-Optimal / Evolutionary |

*Where $n = 	ext{number of cities}$, $g = 	ext{generations}$, and $p = 	ext{population size}$.*

---

## 📂 Project Architecture

The workspace is split into three foundational script layers:

1. **`main.py` (Core GUI Visualizer):** - A fully featured Desktop GUI built using **Tkinter** and **Matplotlib**.
   - Allows users to generate custom city maps dynamically, adjust node counts, step through algorithm visual paths, and review comparative performance curves on the fly.
2. **`performance_test.py` (Analytical Benchmarking Suite):**
   - A standalone benchmarking script meant for rigorous runtime tracking.
   - Evaluates performance thresholds across scaling node profiles and outputs comparative data logs along with trend charts.
3. **`simple_tsp_demo.py` (Core Logic Demo):**
   - A lightweight, clean CLI tool focused strictly on showing core array states and fundamental algorithm mechanics without front-end overhead.

---

## 🛠️ Installation & Setup

### Prerequisites
Ensure you have Python 3.x configured. This project relies on standard mathematical libraries along with graphical drawing modules.

### Step 1: Install Dependencies
Run the following command to grab the required libraries (`matplotlib` and `numpy`):
```bash
pip install matplotlib numpy
```
*(Note: `tkinter` comes pre-packaged with default standard Python installations).*

### Step 2: Run the Main Application
To spin up the primary interactive interface dashboard:
```bash
python main.py
```

### Step 3: Run the Performance Benchmarks
To run scaling tests and evaluate algorithm growth metrics:
```bash
python performance_test.py
```

---

## 🎮 Features & How to Use the Visualizer

1. **Generate Map Bounds:** Open the interface and use the input configuration panel to define your target city node counts ($N$). Click **"Generate Cities"** to deploy random coordinate vectors.
2. **Select & Solve:** Choose an algorithm from the selection menu (e.g., *Greedy NN*, *MST Approx*, *Genetic Algorithm*, or *Brute Force* for $N \le 10$).
3. **Analyze Trends:** Navigate to the **Performance Analysis Dashboard** inside the application to see real-time plots contrasting absolute execution latencies against problem complexity limits.

---

## 📜 License
This project is open-source and free to adapt for educational algorithm reviews, course analysis, or performance tuning experiments!
