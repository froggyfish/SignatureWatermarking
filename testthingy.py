import torch
import numpy as np
import matplotlib.pyplot as plt

# Load tensors
latent = torch.load('decode_testing/0_latent.pt', map_location='cpu')
codeword = torch.load('codeword.pt', map_location='cpu')

# Flatten tensors if needed
def flatten(t):
    return t.view(-1) if hasattr(t, 'view') else t.reshape(-1)

latent_flat = flatten(latent)
codeword_flat = flatten(codeword)

# Ensure same length
min_len = min(len(latent_flat), len(codeword_flat))
latent_flat = latent_flat[:min_len]
codeword_flat = codeword_flat[:min_len]

# 1. Sign similarity
sign_latent = torch.sign(latent_flat[1000:1500])
sign_codeword = torch.sign(codeword_flat[1000:1500])
same_sign = (sign_latent == sign_codeword)
percent_same_sign = same_sign.float().mean().item() * 100
print(f"Percent of indices with same sign: {percent_same_sign:.2f}%")
print(latent_flat[750:780])
print(codeword_flat[750:780])

# 2. Value similarity (absolute difference)
abs_diff = torch.abs(latent_flat - codeword_flat)
avg_abs_diff = abs_diff.mean().item()
print(f"Average absolute difference: {avg_abs_diff:.4f}")

# 3. Burst detection using moving average of sign differences
window_size = 10  # You can adjust this
sign_diff = (~same_sign).float().numpy()
moving_avg = np.convolve(sign_diff, np.ones(window_size)/window_size, mode='valid')

print(f"Max moving average of sign difference (burstiness): {moving_avg.max():.2f}")

# Optional: Plot the moving average to visualize bursts
plt.figure(figsize=(14,4))
plt.plot(moving_avg)
plt.title('Moving Average of Sign Differences (Burst Detection)')
plt.xlabel('Index')
plt.ylabel(f'Fraction of sign differences (window={window_size})')
plt.show()
