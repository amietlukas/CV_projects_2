import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image==0.18.3` to install skimage, if you haven't done so.
# If you use scikit-image>=0.19, you will need to replace the `multichannel=True` argument with `channel_axis=-1`
# for the `skimage.transform.rescale` function
from skimage import io, color
from skimage.transform import rescale

# TO CORRECT THE PATH
from pathlib import Path
data_dir = Path(__file__).resolve().parent


def distance(x, X):
    # X shape (H*W, 3)
    dist = []
    # iterate over all poits
    for point in X:
        norm = torch.linalg.norm(point - x)
        dist.append(norm)
    dist = torch.stack(dist) # convert list to tensor
    return dist



def distance_batch(X, Y):
    # X: (N, D) -> (N, 1, D)
    # Y: (M, D) -> (1, M, D)
    diff = X[:, None, :] - Y[None, :, :]
    dist = torch.linalg.norm(diff, dim=2) # across D
    return dist



def gaussian(dist, bandwidth):
    # gaussian kernel, only the non-const part
    weight = torch.exp(-1/(2*bandwidth**2) * (dist**2))
    return weight



def update_point(weight, X):
    weight = torch.as_tensor(weight, dtype=X.dtype)
    num = torch.zeros_like(X[0])
    den = weight.sum()
    for i in range(X.shape[0]):
        num += weight[i] * X[i]
    return num / den
    
    

def update_point_batch(weight, X):
    weight = torch.as_tensor(weight, dtype=X.dtype)
    num = torch.sum((weight[:, :, None]) * X[None, :, :], dim=1)
    den = torch.sum(weight, axis=1)[:, None]

    updated_points =  num / den
    
    return updated_points



def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_



def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    dist = distance_batch(X, X)
    weight = gaussian(dist, bandwidth)
    X_ = update_point_batch(weight, X)
    return X_


def meanshift(X):
    X = X.clone()
    for _ in range(20):
        X = meanshift_step(X)   # slow implementation =======> ca 1094.31s
        #X = meanshift_step_batch(X)   # fast implementation =======> ca 1.11s
    return X



scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread(data_dir / 'cow.jpg'), scale, channel_axis=-1) # HAD TO CHANGE TO LOAD IMAGE RIGHT
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load(data_dir / 'colors.npz')['colors'] # HAD TO CHANGE FOR RIGHT PATH
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave(data_dir / 'result.png', result_image) # HAD TO CHANGE FOR RIGHT PATH
