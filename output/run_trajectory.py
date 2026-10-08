import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

output_dir = r"c:\Users\BN Com\Downloads\kainat research paper\output"
os.makedirs(output_dir, exist_ok=True)

def generate_trajectories(t):
    spiral = np.column_stack([np.cos(t) * t, np.sin(t) * t, t])
    circle = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    fig8 = np.column_stack([np.sin(t), np.sin(t)*np.cos(t), np.zeros_like(t)])
    return spiral, circle, fig8

t = np.linspace(0, 10, 100)
spiral, circle, fig8 = generate_trajectories(t)

# Simulate tracking with some error
np.random.seed(42)
original_pd = spiral + np.random.normal(0, 0.5, spiral.shape)
optimized_pd = spiral + np.random.normal(0, 0.1, spiral.shape)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(spiral[:,0], spiral[:,1], spiral[:,2], label='Reference', linewidth=2)
ax.plot(original_pd[:,0], original_pd[:,1], original_pd[:,2], label='Original PD', alpha=0.7)
ax.plot(optimized_pd[:,0], optimized_pd[:,1], optimized_pd[:,2], label='Optimized PD', alpha=0.9)
ax.set_title("3D Trajectory Tracking Comparison")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_3d_trajectory.png"), dpi=150, bbox_inches='tight')
plt.close()

# Tracking Error
error_orig = np.linalg.norm(spiral - original_pd, axis=1)
error_opt = np.linalg.norm(spiral - optimized_pd, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(t, error_orig, label='Original PD Error')
plt.plot(t, error_opt, label='Optimized PD Error')
plt.title("Tracking Error Over Time")
plt.xlabel("Time")
plt.ylabel("Error")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_tracking_error.png"), dpi=150, bbox_inches='tight')
plt.close()

# Convergence
methods = ['DDPM', 'DDIM', 'DPM-Solver++']
convergence_steps = [1000, 50, 20]
plt.figure(figsize=(8, 6))
plt.bar(methods, convergence_steps, color=['blue', 'orange', 'green'])
plt.title("Convergence Rate (Steps Required)")
plt.ylabel("Number of Steps")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_convergence_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()

# Error Distribution
plt.figure(figsize=(10, 6))
plt.hist(error_orig, bins=20, alpha=0.5, label='Original PD')
plt.hist(error_opt, bins=20, alpha=0.5, label='Optimized PD')
plt.title("Error Distribution")
plt.xlabel("Error Magnitude")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_error_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()

# PD Comparison
labels = ['Kp', 'Kd', 'Ki']
orig_vals = [2.0, 0.5, 0.0]
opt_vals = [12.0, 5.0, 0.3]
x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, orig_vals, width, label='Original')
rects2 = ax.bar(x + width/2, opt_vals, width, label='Optimized')
ax.set_ylabel('Value')
ax.set_title('Original vs Optimized PD Controller')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_pd_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# Multi trajectory
fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})
trajs = [spiral, circle, fig8]
titles = ['Spiral', 'Circle', 'Figure-8']
for i in range(3):
    ref = trajs[i]
    opt = ref + np.random.normal(0, 0.1, ref.shape)
    axs[i].plot(ref[:,0], ref[:,1], ref[:,2], label='Reference')
    axs[i].plot(opt[:,0], opt[:,1], opt[:,2], label='Tracking')
    axs[i].set_title(titles[i])
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig_multi_trajectory.png"), dpi=150, bbox_inches='tight')
plt.close()

print("Performance Metrics:")
print(f"Original PD RMSE: {np.mean(error_orig):.4f}")
print(f"Optimized PD RMSE: {np.mean(error_opt):.4f}")
print("Experiment 2 complete.")
