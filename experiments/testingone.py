from skimage import data
from skimage import io
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
import numpy as np
import itertools
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

image = io.imread("dipper1.jpg")
height = len(image[0])
width = len(image)
# image = data.hubble_deep_field()[0:500, 0:500]
image_gray = rgb2gray(image)

blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)

# Compute radii in the 3rd column.
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
# filter blogs greater then 1 radius
blobs_log =  blobs_log[blobs_log[:,2]>1,:]

blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
# filter blogs greater then 2 radius
blobs_dog =  blobs_dog[blobs_dog[:,2]>2,:]

blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)
# filter blogs greater then 4 radius
blobs_doh =  blobs_doh[blobs_doh[:,2]>4,:]

blobs_list = [blobs_log, blobs_dog, blobs_doh]
colors = ['yellow', 'lime', 'red']
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
sequence = zip(blobs_list, colors, titles)

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True,
                         subplot_kw={'adjustable': 'box-forced'})
plt.tight_layout()

axes = axes.ravel()
for blobs, color, title in sequence:
    ax = axes[0]
    axes = axes[1:]
    ax.set_title(title)
    ax.imshow(image, interpolation='nearest')
    ax.set_axis_off()
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax.add_patch(c)
    points = blobs[:,:-1]
    # for x in np.nditer(points, op_flags=['readwrite']):
    #     x = [height - x[0], x[1]]

    print(points)
    ax.plot(*zip(*itertools.chain.from_iterable(itertools.combinations(points, 2))),color='brown', marker='o')

# a = np.sort(blobs_log, key=lambda row: row[1])
# arr = blobs_log[blobs_log[:,2].argsort()]
# print(blobsr_log[blobs_log[:,2]>1,:])
# print(blobs_dog[blobs_dog[:,2]>1,:])
# print(blobs_doh[blobs_doh[:,2]>1,:])
plt.show()

# https://snipt.net/Miki/sort-array-by-nth-column-in-numpypython/
# http://stackoverflow.com/questions/6834483/how-do-you-create-line-segments-between-two-points