import numpy as np
import matplotlib.pyplot as plt

# Parameters
beta = 0.1
steps = 5
dim = 2  # 2D points
x = np.zeros((steps + 1, dim))  # store x0 to x5
means = np.zeros_like(x)

# Initial point x0
x[0] = np.array([1.0, 1.0])

# Diffusion process: sample x_t ~ N((1 - beta)x_{t-1}, beta * I)
for t in range(1, steps + 1):
    mean = (1 - beta) * x[t - 1]
    noise = np.random.normal(0, np.sqrt(beta), size=dim)
    x[t] = mean + noise
    means[t] = mean

# Plot the results
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x[:, 0], x[:, 1], 'o-', label='Sampled points (x_t)')
ax.quiver(x[:-1, 0], x[:-1, 1], means[1:, 0] - x[:-1, 0], means[1:, 1] - x[:-1, 1],
          angles='xy', scale_units='xy', scale=1, color='orange', alpha=0.5, label='Mean direction')

for i in range(steps + 1):
    ax.text(x[i, 0] + 0.01, x[i, 1] + 0.01, f"x{i}")

ax.set_title("Diffusion Process: Sampling xₜ from q(xₜ | xₜ₋₁)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
plt.axis('equal')
plt.show()
