import torch
from pathlib import Path

input_folder = Path("tampered_latents")

for latent in input_folder.iterdir():

    latent1 = torch.load('history_noises/initial_noise.pt')  # e.g., shape might be (1, 4, 64, 64)
    latent2 = torch.load(f'tampered_latents/{latent.name}')

    # Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(latent1 - latent2)).item()

    # Mean Squared Error (MSE)
    mse = torch.mean((latent1 - latent2)**2).item()

    # Cosine similarity (flatten tensors first)
    cos_sim = torch.nn.functional.cosine_similarity(latent1.flatten(), latent2.flatten(), dim=0).item()

    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"Cosine Similarity: {cos_sim:.6f}")

    import matplotlib.pyplot as plt

    # For visualization, pick the first channel or mean over channels
    def prepare_image(tensor):
        if tensor.ndim == 4:
            img = tensor[0, 0].cpu().numpy()  # first sample, first channel
        elif tensor.ndim == 3:
            img = tensor[0].cpu().numpy()
        else:
            img = tensor.cpu().numpy()
        
        # Normalize to [0, 1] for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img

    img1 = prepare_image(latent1)
    img2 = prepare_image(latent2)
    diff = prepare_image(torch.abs(latent1 - latent2))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Latent 1')
    plt.imshow(img1, cmap='viridis')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Latent 2')
    plt.imshow(img2, cmap='viridis')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Absolute Difference')
    plt.imshow(diff, cmap='inferno')
    plt.axis('off')

    plt.show()
