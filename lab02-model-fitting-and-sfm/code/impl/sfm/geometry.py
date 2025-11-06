import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # TODO
  # Normalize coordinates (to points on the normalized image plane)
  # normalized camera coordinates 
  K_inv = np.linalg.inv(K) # 3x3
  hom_kps1 = MakeHomogeneous(im1.kps.T) # 3xN each column (u, v, 1)
  hom_kps2 = MakeHomogeneous(im2.kps.T) 
  normalized_kps1 = (K_inv @ hom_kps1).T # Nx3 each row (x, y, z)
  normalized_kps2 = (K_inv @ hom_kps2).T 
  normalized_kps1 /= normalized_kps1[:, [2]] # Nx3 each row (x, y, 1) -> force last coord = 1
  normalized_kps2 /= normalized_kps2[:, [2]]
  
  # Assemble constraint matrix as equation 1
  constraint_matrix = np.zeros((matches.shape[0], 9))
  for row, (idx1, idx2) in enumerate(matches): # inx1, idx2 are indices of matched keypoints in im1 and im2
    # TODO
    # get the normalized keypoints for the current match
    x1, y1, z1 = normalized_kps1[idx1]  # (x1, y1, 1)
    x2, y2, z2 = normalized_kps2[idx2]  # (x2, y2, 1)

    constraint_matrix[row] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2,x1, y1, 1]       
    
   
  # Solve for the nullspace of the constraint matrix
  _, _, vt = np.linalg.svd(constraint_matrix) # svd on constraint matrix
  E_hat = vt[-1].reshape(3, 3) # last row of V^T is the solution

  # TODO
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  U, S, Vt = np.linalg.svd(E_hat)
  E = U @ np.diag([1, 1, 0]) @ Vt

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.  
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i, 0], :]  # image 1 point
    kp1 = kp1[:, None]  # make it a column vector
    kp2 = normalized_kps2[matches[i, 1], :]  # image 2 point
    kp2 = kp2[:, None]  # make it a column vector
    assert abs(kp2.T @ E @ kp1) < 0.01

  return E



def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols



def TriangulatePoints(K, im1, im2, matches):
    R1, t1 = im1.Pose()
    R2, t2 = im2.Pose()
    P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
    P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

    # Ignore matches that already have a triangulated point
    new_matches = np.zeros((0, 2), dtype=int)
    used_im1 = set()
    used_im2 = set()

    num_matches = matches.shape[0]
    for i in range(num_matches):
        p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
        p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
        kp1_idx = matches[i, 0]
        kp2_idx = matches[i, 1]
        if (
            p3d_idx1 == -1 and
            p3d_idx2 == -1 and
            kp1_idx not in used_im1 and
            kp2_idx not in used_im2
        ):
            new_matches = np.append(new_matches, matches[[i]], 0)
            used_im1.add(kp1_idx)
            used_im2.add(kp2_idx)

    num_new_matches = new_matches.shape[0]

    points3D = np.zeros((num_new_matches, 3))

    reproj_errors = []
    for i in range(num_new_matches):
        kp1 = im1.kps[new_matches[i, 0], :]
        kp2 = im2.kps[new_matches[i, 1], :]

        # H & Z Sec. 12.2
        A = np.array([
            kp1[0] * P1[2] - P1[0],
            kp1[1] * P1[2] - P1[1],
            kp2[0] * P2[2] - P2[0],
            kp2[1] * P2[2] - P2[1]
        ])

        _, _, vh = np.linalg.svd(A)
        homogeneous_point = vh[-1]
        points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]

        # Compute reprojection error in both images for filtering
        proj1 = P1 @ np.append(points3D[i], 1)
        proj1 = proj1[:2] / proj1[2]
        proj2 = P2 @ np.append(points3D[i], 1)
        proj2 = proj2[:2] / proj2[2]

        err = np.linalg.norm(proj1 - kp1) + np.linalg.norm(proj2 - kp2)
        reproj_errors.append(err)

    reproj_errors = np.array(reproj_errors)

    # Filter out matches with large reprojection error
    if reproj_errors.size > 0:
        mask = reproj_errors < 5.0
        points3D = points3D[mask]
        im1_corrs = new_matches[:, 0][mask]
        im2_corrs = new_matches[:, 1][mask]
    else:
        im1_corrs = new_matches[:, 0]
        im2_corrs = new_matches[:, 1]

    # We need to keep track of the correspondences between image points and 3D points
    # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
    # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`

    # Filter points behind the first camera
    mask = ((R1 @ points3D.T).T + t1)[:, 2] > 0
    im1_corrs = im1_corrs[mask]
    im2_corrs = im2_corrs[mask]
    points3D = points3D[mask]

    # Filter points behind the second camera
    mask = ((R2 @ points3D.T).T + t2)[:, 2] > 0
    im1_corrs = im1_corrs[mask]
    im2_corrs = im2_corrs[mask]
    points3D = points3D[mask]

    return points3D, im1_corrs, im2_corrs



def EstimateImagePose(points2D, points3D, K):
    # We use points in the normalized image plane.
    # This removes the 'K' factor from the projection matrix.
    # We don't normalize the 3D points here to keep the code simpler.
    K_inv = np.linalg.inv(K)
    normalized_points2D = HNormalize((K_inv @ MakeHomogeneous(points2D, 1).T).T, 1)

    constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

    # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
    # (the manifold of proper 3D poses). This is a bit too complicated right now.
    # Just DLT should give good enough results for this dataset.

    # Solve for the nullspace
    _, _, vh = np.linalg.svd(constraint_matrix)
    P_vec = vh[-1, :]
    P = np.reshape(P_vec, (3, 4), order='C')

    # Make sure we have a proper rotation
    u, s, vh = np.linalg.svd(P[:, :3])
    R = u @ vh

    if np.linalg.det(R) < 0:
        R *= -1

    _, _, vh = np.linalg.svd(P)
    C = np.copy(vh[-1, :])

    t = -R @ (C[:3] / C[3])

    return R, t



def TriangulateImage(K, image_name, images, registered_images, matches):
    # Loop over all registered images and triangulate new points with the new image.
    # Make sure to keep track of all new 2D-3D correspondences, also for the registered images
    corrs = {}
    num_points = 0
    all_image_corr = np.empty(0, dtype=int)
    points3D = np.zeros((0, 3))
    used_new_image_kps = set()

    for reg_name in registered_images:
        # Triangulate points between the new image and the registered image
        reg_matches = GetPairMatches(reg_name, image_name, matches)
        points, reg_corr, im_corr = TriangulatePoints(K, images[reg_name], images[image_name], reg_matches)
        if points.shape[0] == 0:
            continue

        # Prevent triangulating the same keypoint of the new image multiple times
        mask = np.asarray([kp not in used_new_image_kps for kp in im_corr], dtype=bool)
        if not np.any(mask):
            continue

        points = points[mask]
        reg_corr = np.asarray(reg_corr, dtype=int)[mask]
        im_corr = np.asarray(im_corr, dtype=int)[mask]

        used_new_image_kps.update(im_corr.tolist())
        num_new_points = points.shape[0]

        # Append the new points to the global point array
        points3D = np.concatenate([points3D, points], axis=0)
        corrs[reg_name] = (reg_corr, num_points, num_points + num_new_points)

        # Add the correspondences to the image
        all_image_corr = np.concatenate([all_image_corr, im_corr], axis=0)
        num_points += num_new_points

    # Add the correspondences of the new image
    corrs[image_name] = (np.asarray(all_image_corr, dtype=int), 0, num_points)
    return points3D, corrs
  
