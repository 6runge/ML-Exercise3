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

print(str(len(images)) + " images loaded\n")
#imgplot = plt.imshow(images[0][0])
#plt.show()

#### Obtain feature vectors
## helper functions for the featurizers
def mean_of_channel(image, channel):
    total = 0
    number_of_pixels = 0
    for row in image:
        for pixel in row:
            total += pixel[channel]
            number_of_pixels += 1
    return total / number_of_pixels

## each featurizer "extracts" a feature from an image, specific featurizers can be (de-)activated by (un-)commenting
featurizers = [
    (lambda image: mean_of_channel(image, 0)),
    (lambda image: mean_of_channel(image, 1)),
    (lambda image: mean_of_channel(image, 2)),
]
Xs = []
for image, label in images:
    x = []
    for featurizer in featurizers:
        x.append(featurizer(image))
    x.append(label)
    Xs.append(x)

print("Example feature vector:")
print(Xs[random.randint(0, len(Xs))])

#### Initialize your parameters
# TODO

#### Implement Gaussian Discriminant analysis
# TODO