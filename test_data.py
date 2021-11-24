import numpy as np
from PIL import Image

rollout = np.load("data/rollouts/ClothFold_rollout_0.npz")

print(f"Shape: {rollout['observations'].shape}")

print(rollout['observations'][0])

# save an observations
pil_img = Image.fromarray(rollout['observations'][0])
pil_img.show()
