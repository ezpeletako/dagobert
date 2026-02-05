import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

root = tk.Tk()
root.title("GUI Test")

# Simple Matplotlib figure
fig, ax = plt.subplots()
ax.plot([0, 1, 2], [0, 1, 4])
ax.set_title("Test Plot")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill="both", expand=1)

root.mainloop()

