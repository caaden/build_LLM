import torch
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Panel 1: softmax as a function of one input (others fixed at 0) ---
x_vals = np.linspace(-5, 5, 300)
# softmax([x, 0]) — probability assigned to first element as x varies
softmax_curve = np.exp(x_vals) / (np.exp(x_vals) + np.exp(0))

ax1 = axes[0]
ax1.plot(x_vals, softmax_curve, color='steelblue', linewidth=2)
ax1.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='y = 0.5')
ax1.axvline(0.0, color='gray', linestyle='--', linewidth=0.8, label='x = 0')
ax1.set_title("Softmax curve\nsoftmax([x, 0])[0]  vs  x", fontsize=12)
ax1.set_xlabel("x (score for token 1)")
ax1.set_ylabel("Attention weight (probability)")
ax1.set_ylim(-0.05, 1.05)
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Panel 2: actual attention scores vs softmax weights from the exercise ---
token_labels = ["Your\n(x1)", "journey\n(x2)", "starts\n(x3)", "with\n(x4)", "one\n(x5)", "step\n(x6)"]
attn_scores = torch.tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
softmax_weights = torch.softmax(attn_scores, dim=0).numpy()
naive_weights = (attn_scores / attn_scores.sum()).numpy()
raw_scores = attn_scores.numpy()

x = np.arange(len(token_labels))
width = 0.25

ax2 = axes[1]
bars1 = ax2.bar(x - width, raw_scores / raw_scores.sum(),  width, label='Sum-norm (line 28)', color='salmon', alpha=0.85)
bars2 = ax2.bar(x,         softmax_weights,                 width, label='Softmax (line 38)',  color='steelblue', alpha=0.85)
ax2.set_title("Attention weights for 'journey' query\nsum-norm vs softmax", fontsize=12)
ax2.set_xticks(x - width/2)
ax2.set_xticklabels(token_labels, fontsize=9)
ax2.set_ylabel("Weight (sums to 1)")
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("softmax_plot.png", dpi=130)
print("Saved to softmax_plot.png")
