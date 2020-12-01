import matplotlib.pyplot as plt
import matplotlib.image as img
from pathlib import Path
import random
import math
import numpy as np

#### Options
## 1 path per label
paths = ["./img/subset/mask", "./img/subset/no_mask"]
labels = ["masked", "unmasked"]

#### Load the images
images = []
for i in range(len(paths)):
    path = Path(paths[i]).rglob("*.png")
    files = [x for x in path]
    for f in files:
        images.append((img.imread(f), i))

print(images[0])
imgplot = plt.imshow(images[0][0])
plt.show()

#### Obtain feature vectors


#### Initialize your parameters


#### Implement Gaussian Discriminant analysis


####
