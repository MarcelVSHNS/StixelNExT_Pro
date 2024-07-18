import torch
import numpy as np
from einops import rearrange

# Beispiel-Tensor (ersetze dies mit deinem tatsächlichen Tensor)
tensor = torch.randn(4, 4, 12, 240)

# Umwandlung in NumPy-Array
prediction = tensor.numpy()

for batch in prediction:
    # print(f"Batch1: {batch.shape}")
    colums = rearrange(batch, "a n w -> w n a")
    for u in range(len(colums)):
        # print(f"Col1: {column.shape}")
        for candidate in colums[u]:
            # print(f"candidate1: {candidate.shape}")
            print(f"val_{u}: {candidate[0]}")
            break
        break
    break
