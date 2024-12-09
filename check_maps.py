# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils import load_obj

# %%
data_dir = Path("data_exp")
env_name = "CarRacing-v3"
data_files = sorted(data_dir.glob(f"*.pkl"))
for di, data_file in enumerate(data_files):
    data = load_obj(data_file)
    s = data[12][0]
    # plt.imshow(s.squeeze(0).numpy(), cmap="gray")
    plt.imshow(s[0], cmap="gray")
    plt.show()
    if di > 10:
        break
# %%
