import numpy as np

# Find (unique) 2D-3D correspondences from 2D-2D correspondences
def Find2D3DCorrespondences(image_name, images, matches, registered_images):
  assert(image_name not in registered_images)

  image_kp_idxs = []
  p3D_idxs = []
  for other_image_name in registered_images:
    other_image = images[other_image_name]
    pair_matches = GetPairMatches(image_name, other_image_name, matches)

    for i in range(pair_matches.shape[0]):
      p3D_idx = other_image.GetPoint3DIdx(pair_matches[i,1])
      if p3D_idx > -1:
        p3D_idxs.append(p3D_idx)
        image_kp_idxs.append(pair_matches[i,0])

  print(f'found {len(p3D_idxs)} points, {np.unique(np.array(p3D_idxs)).shape[0]} unique points')

  # Remove duplicated correspondences
  _, unique_idxs = np.unique(np.array(p3D_idxs), return_index=True)
  image_kp_idxs = np.array(image_kp_idxs)[unique_idxs].tolist()
  p3D_idxs = np.array(p3D_idxs)[unique_idxs].tolist()
  
  return image_kp_idxs, p3D_idxs



# Make sure we get keypoint matches between the images in the order that we requested
def GetPairMatches(im1, im2, matches):
  if im1 < im2:
    return matches[(im1, im2)]
  else:
    return np.flip(matches[(im2, im1)], 1)



# Update the reconstruction with the new information from a triangulated image
def UpdateReconstructionState(new_points3D, corrs, points3D, images):
  
  if new_points3D.size == 0:
    return points3D, images

  offset = points3D.shape[0]
  points3D = np.concatenate([points3D, new_points3D], axis=0)

  for im_name, (corr, start, end) in corrs.items():
    if end <= start:
      continue

    kp_idxs = np.asarray(corr, dtype=int).tolist()
    global_idxs = (offset + np.arange(start, end, dtype=int)).tolist()
    assert len(kp_idxs) == len(global_idxs)

    images[im_name].Add3DCorrs(kp_idxs, global_idxs)

  return points3D, images
