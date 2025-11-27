import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def img_show(img, dpi=100):
    plt.figure(figsize=(img.shape[1] / dpi, img.shape[0] / dpi), dpi=dpi)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


def sobel_gradients(gray: torch.Tensor):
    H, W = gray.shape

    # TODO: Compute Sobel gradients. (10pts)

    # Define kernels manually (do not call cv2.Sobel).
    kernel_horizontal = torch.tensor([[-1., 0., 1.], 
                                      [-2., 0., 2.], 
                                      [-1., 0., 1.]], dtype=torch.float32)
    kernel_vertical = torch.tensor([[-1., -2., -1.], 
                                    [0., 0., 0.], 
                                    [1., 2., 1.]], dtype=torch.float32)
    
    # reshape kernels to fit conv2d (Cout, Cin, 3, 3)
    kx = kernel_horizontal.view(1, 1, 3, 3)
    ky = kernel_vertical.view(1, 1, 3, 3)
    # reshape kernels to fit conv2d (Cout, Cin, H, W)
    gray_batch_channel = gray.view(1, 1, H, W)

    # OpenCV applies mirror padding 1 pixel
    gray_padded = F.pad(gray_batch_channel, pad=(1, 1, 1, 1), mode='reflect')

    gradient_x = F.conv2d(gray_padded, kx)
    gradient_y = F.conv2d(gray_padded, ky) 

    # remove batch and channel dimensions
    gradient_x = gradient_x.view(H, W)
    gradient_y = gradient_y.view(H, W)

    # Compute L2 magnitude and angle (in radians):
    magnitude = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)
    angle = torch.atan2(gradient_y, gradient_x) # [-pi, pi]

    assert gradient_x.shape == (H, W)
    assert gradient_y.shape == (H, W)
    assert magnitude.shape == (H, W)
    assert angle.shape == (H, W)
    assert angle.min() >= -np.pi and angle.max() <= np.pi

    return gradient_x, gradient_y, magnitude, angle



def non_max_suppression(magnitude: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    H, W = magnitude.shape
    assert magnitude.shape == (H, W)
    assert angle.shape == (H, W)

    # check wwether pixel is local maximum.

    # TODO: Perform non-maximum suppression. (20pts)
    magnitude_after_nms = torch.zeros_like(magnitude) # same shape as maginitude
    # convert to [0, 180)
    angle_deg = (torch.rad2deg(angle) + 180.0) % 180.0
    # leave out boarder pixels
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            magnitude_current = magnitude[y, x]
            angle_current = angle_deg[y, x].item()
            # determine the two neighboring pixels to compare
            # case 1, 0 degrees
            if (0 <= angle_current <= 22.5) or (157.5 <= angle_current < 180):
                neighbor1 = magnitude[y, x+1]
                neighbor2 = magnitude[y, x-1]
            # case 2, 0 degrees
            elif (22.5 < angle_current <= 67.5):
                neighbor1 = magnitude[y-1, x-1]
                neighbor2 = magnitude[y+1, x+1]
            # case 3, 0 degrees
            elif (67.5 < angle_current <= 112.5):
                neighbor1 = magnitude[y+1, x]
                neighbor2 = magnitude[y-1, x]
            else: # 112.5 <= angle_current < 157.5
                # neighbor1 = magnitude[y-1, x-1]
                # neighbor2 = magnitude[y+1, x+1]
                neighbor1 = magnitude[y-1, x+1]
                neighbor2 = magnitude[y+1, x-1]
            
            # keep only local maximum
            if neighbor1 <= magnitude_current and neighbor2 <= magnitude_current:
                magnitude_after_nms[y, x] = magnitude_current
            else:
                magnitude_after_nms[y, x] = 0.0


    assert magnitude_after_nms.shape == (H, W)
    assert magnitude_after_nms.sum() <= magnitude.sum()

    return magnitude_after_nms



def double_threshold_and_hysteresis(nms: torch.Tensor, low: float, high: float) -> torch.Tensor:
    H, W = nms.shape
    assert nms.shape == (H, W)
    assert low >= 0
    assert high >= 0
    assert low <= high

    # TODO: Run DFS to propagate strong edges to weak edges. (20pts)
    strong_mask = nms >= high # sure edge
    weak_mask = (nms >= low) & (nms < high) # maybe edge
    
    is_edge = torch.zeros((H, W), dtype=torch.bool)

    # initialize stack with all strong pixels
    ys, xs = torch.nonzero(strong_mask, as_tuple=True)
    stack = [(int(y), int(x)) for y, x in zip(ys, xs)]

    # all strong pixels are edges
    for y, x in stack:
        is_edge[y, x] = True
    
    # look at each strong pixel, and see if it is connected to weak one. if yes, weak pixel is also edge pixel (8 neighbors)
    while stack:
        y, x = stack.pop() # strong pixel

        # look at 8 neighbors
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                
                # skip itself
                if dy == 0 and dx == 0:
                    continue
                
                neighbor_y = y + dy
                neighbor_x = x + dx
                # check bounds
                if 0 <= neighbor_y < H and 0 <= neighbor_x < W:
                    # if neighbor is weak pixel and not yet marked as edge
                    if weak_mask[neighbor_y, neighbor_x] and not is_edge[neighbor_y, neighbor_x]:
                        is_edge[neighbor_y, neighbor_x] = True
                        stack.append((neighbor_y, neighbor_x)) # add new strong pixel to stack

    assert is_edge.shape == (H, W)
    assert is_edge.dtype == torch.bool

    return is_edge


def canny(gray: torch.Tensor, low: float, high: float) -> torch.Tensor:
    gradient_x, gradient_y, magnitude, angle = sobel_gradients(gray)
    nms = non_max_suppression(magnitude, angle)
    edges = double_threshold_and_hysteresis(nms, low, high)
    return edges.float()


def main(
        image_path,
        logs_dir="results",
        threshold_low=100,
        threshold_high=200,
        blur_kernelsize=5,
        blur_sigma=1.0,
        show_images=False,
):
    os.makedirs(logs_dir, exist_ok=True)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray_np = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocess: Gaussian blur to reduce noise
    gray_blur_numpy = cv2.GaussianBlur(gray_np, (blur_kernelsize, blur_kernelsize), sigmaX=blur_sigma)
    gray_blur_torch = torch.from_numpy(gray_blur_numpy).float()

    # Canny edge detection: our implementation vs OpenCV
    print("Running Canny edge detection...")
    edges_ours = canny(gray_blur_torch, low=threshold_low, high=threshold_high).cpu().numpy()
    edges_opencv = cv2.Canny(gray_blur_numpy.astype(np.uint8), threshold_low, threshold_high, 3, L2gradient=True)

    # Log results
    if logs_dir is not None:
        print(f"Saving results to: {os.path.abspath(logs_dir)}")
        cv2.imwrite(os.path.join(logs_dir, "edges_ours.png"), (edges_ours * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(logs_dir, "edges_opencv.png"), edges_opencv)
    if show_images:
        img_show(edges_ours, dpi=100)
        img_show(edges_opencv, dpi=100)

    # Compute IoU
    ours_bin = (edges_ours > 0).astype(np.uint8)
    cv_bin = (edges_opencv > 0).astype(np.uint8)
    inter = np.logical_and(ours_bin == 1, cv_bin == 1).sum()
    union = np.logical_or(ours_bin == 1, cv_bin == 1).sum()
    iou = inter / union if union > 0 else 0.0
    print(f"IoU: {iou * 100:>6.2f}% (low={threshold_low}, high={threshold_high}, img={image_path})\n")


if __name__ == "__main__":
    for img_file, threshold_low, threshold_high, blur_sigma in [
        ("data/lenna.png", 100, 200, 1.0),
        ("data/lenna.png", 200, 200, 1.0),
    ]:
        main(
            image_path=img_file,
            logs_dir=f"results/canny_low{threshold_low}_high{threshold_high}",
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            show_images=False,
        )
