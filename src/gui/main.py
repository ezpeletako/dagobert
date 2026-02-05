import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from dagobert.vector_replicator import VectorReplicator
from dagobert.rng import make_rng
from dagobert.live import evolve_generator

# ----------------------------
# GUI setup
# ----------------------------
root = tk.Tk()
root.title("Dagobert Evolution Live Demo")

# Matplotlib figure
fig, ax = plt.subplots(figsize=(6, 4))
line, = ax.plot([], [], label="Best Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Fitness")
ax.set_title("Live Evolution of Best Fitness")
ax.legend()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# ----------------------------
# Initialize population
# ----------------------------
rng = make_rng(42)
pop_size = 50
genome_size = 10
pop = [VectorReplicator(rng.normal(size=genome_size)) for _ in range(pop_size)]
generations = 30
best_history = []

# ----------------------------
# Control panel
# ----------------------------
control_frame = ttk.Frame(root)
control_frame.pack(side=tk.BOTTOM, fill=tk.X)

start_btn = ttk.Button(control_frame, text="Start")
start_btn.pack(side=tk.LEFT, padx=5, pady=5)

sigma_label = ttk.Label(control_frame, text="Sigma:")
sigma_label.pack(side=tk.LEFT, padx=5)
sigma_var = tk.DoubleVar(value=0.1)
sigma_entry = ttk.Entry(control_frame, textvariable=sigma_var, width=5)
sigma_entry.pack(side=tk.LEFT)

elite_label = ttk.Label(control_frame, text="Elite Fraction:")
elite_label.pack(side=tk.LEFT, padx=5)
elite_var = tk.DoubleVar(value=0.05)
elite_entry = ttk.Entry(control_frame, textvariable=elite_var, width=5)
elite_entry.pack(side=tk.LEFT)

# ----------------------------
# Evolution loop
# ----------------------------
def run_evolution():
    global pop, best_history
    line.set_xdata([])
    line.set_ydata([])
    best_history = []

    gen = evolve_generator(
        pop,
        rng,
        generations=generations,
        sigma=sigma_var.get(),
        elite_frac=elite_var.get()
    )

    def update():
        nonlocal gen
        try:
            pop, fitnesses = next(gen)
            best = fitnesses.max()
            best_history.append(best)
            line.set_xdata(range(len(best_history)))
            line.set_ydata(best_history)
            ax.relim()
            ax.autoscale_view()
            canvas.draw()
            root.after(100, update)  # schedule next update in 100 ms
        except StopIteration:
            print("Evolution finished!")

    update()

start_btn.config(command=run_evolution)

# ----------------------------
# Start GUI loop
# ----------------------------
root.mainloop()
