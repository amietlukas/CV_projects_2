import os

import numpy as np
import torch
from tqdm import tqdm

from raft_core.raftnet import Raftnet
from visualizer_mp4 import Visualizer


class CoTracker3Online(torch.nn.Module):
    def __init__(self, model_name="cotracker3_online", grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        self.cotracker = torch.hub.load("facebookresearch/co-tracker", model_name)
        self.cotracker.eval()

    def forward(self, rgbs, query_points):
        T, _, H, W = rgbs.shape
        N, _ = query_points.shape

        assert rgbs.shape == (T, 3, H, W)
        assert rgbs.dtype == torch.uint8
        assert query_points.shape == (N, 3)

        # Forward pass: https://github.com/facebookresearch/co-tracker/blob/82e02e8029753ad4ef13cf06be7f4fc5facdda4d/cotracker/predictor.py#L230
        self.cotracker(
            video_chunk=rgbs[None].float(),
            queries=query_points[None].float(),
            grid_size=self.grid_size,
            is_first_step=True,
        )
        for t in tqdm(range(0, T - self.cotracker.step, self.cotracker.step), desc="Running CoTracker3 Online..."):
            pred_tracks, pred_visibility = self.cotracker(video_chunk=rgbs[None, t: t + self.cotracker.step * 2])
        pred_tracks = pred_tracks.squeeze(0)
        pred_visibility = pred_visibility.squeeze(0)

        assert pred_tracks.shape == (T, N, 2)
        assert pred_visibility.shape == (T, N)

        return pred_tracks, pred_visibility


def compute_tracking_accuracy(
        pred_tracks: torch.Tensor,
        gt_tracks: torch.Tensor,
        gt_vis: torch.Tensor,
        query_points: torch.Tensor,
        distance_threshold_px: float,
) -> float:
    T, N, _ = gt_tracks.shape
    assert gt_tracks.shape == (T, N, 2)
    assert gt_vis.shape == (T, N)
    assert pred_tracks.shape == (T, N, 2)
    assert query_points.shape == (N, 3)

    query_points_t = query_points[:, 0].long() # all t0 for each track n 
    query_points_xy = query_points[:, 1:3] # all (x0, y0) for each track n

    # TODO: Compute distance-within-threshold accuracy. (10pts)
    #  - For each track n, only evaluate timesteps t > query_points[n, 0] and where gt_vis is True.
    #  - Count positions where the Euclidean error <= distance_threshold_px.
    #  - Compute accuracy per track, then average over tracks.
   
    acc_per_track = []

    for n in range(N): # iterate over tracks
        t0 = int(query_points_t[n].item())
        #x0, y0 = query_points_xy[n]
        correct_count = 0
        total_count = 0

        for t in range(t0+1, T): # iterate over frames
            if gt_vis[t, n]: # only consider visible points
                gt_x, gt_y = gt_tracks[t, n]
                pred_x, pred_y = pred_tracks[t, n]

                # euclidean distance
                l2 = torch.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2).item()

                # distance threshold check
                if l2 <= distance_threshold_px:
                    correct_count += 1
                
                total_count += 1
        
        acc_this_track = correct_count / total_count if total_count > 0 else 0.0
        acc_per_track.append(acc_this_track)

    avg_acc = sum(acc_per_track) / len(acc_per_track)

    assert 0.0 <= avg_acc <= 1.0
    return avg_acc



class RaftPointTracker:
    """
    Implements a point tracker that uses the RAFT algorithm for optical flow estimation
    from https://arxiv.org/abs/2003.12039. The tracker computes forward and backward flows
    for each frame in a video sequence and uses these to estimate the trajectories of given points.
    """

    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        if self.checkpoint_path is not None and not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Raft checkpoint not found at {self.checkpoint_path}")
        print(f"Loading Raft model from {self.checkpoint_path}")
        self.model = Raftnet(ckpt_name=self.checkpoint_path)
        self.model.eval()

    def forward(self, rgbs, query_points):
        T, _, H, W = rgbs.shape
        N = query_points.shape[0]

        assert rgbs.shape == (T, 3, H, W)
        assert rgbs.dtype == torch.uint8
        assert query_points.shape == (N, 3)

        # TODO: Implement point tracking using RAFT optical flow. (40pts total)
        #  - Compute forward optical flow between consecutive frames using RAFT. (10/40pts)
        #  - For each point, starting from its query position, use the forward flows to
        #    track its position through the video frames. (30/40pts)
        #  - Return the estimated trajectories of all points over all frames.

        trajectories = ...

        assert trajectories.shape == (T, N, 2)

        return trajectories, None


def main(
        datapoint_npz_path="data/horsejump-high.npz",
        logs_dir="results/tracking",
        distance_threshold_px=10.0,
):
    os.makedirs(logs_dir, exist_ok=True)

    # Load datapoint
    data = np.load(datapoint_npz_path, allow_pickle=True)
    rgbs = torch.from_numpy(data["video"]).permute(0, 3, 1, 2)
    gt_tracks = torch.from_numpy(data["traj"])
    gt_vis = torch.from_numpy(data["vis"])
    query_points = torch.from_numpy(data["query_points"])

    T, C, H, W = rgbs.shape
    N = gt_tracks.shape[1]
    assert rgbs.shape == (T, 3, H, W)
    assert gt_tracks.shape == (T, N, 2)
    assert gt_vis.shape == (T, N)
    assert query_points.shape == (N, 3)
    print(f"Loaded sequence {datapoint_npz_path} with {T} frames, {N} points, resolution {W}x{H}.")

    # Unpack query points and make sure they match GT tracks at queried timestep
    query_points_t = query_points[:, 0].long()
    query_points_xy = query_points[:, 1:3]
    assert torch.allclose(gt_tracks[query_points_t, torch.arange(N)], query_points_xy)

    # Visualize GT tracks
    visualizer = Visualizer(save_dir=logs_dir, pad_value=0, fps=30, show_first_frame=0, tracks_leave_trace=-1)
    viz_gt, _ = visualizer.visualize(
        video=rgbs[None],
        tracks=gt_tracks[None],
        visibility=gt_vis[None],
        query_frame=query_points_t[None],
        filename="ground_truth",
        title="Ground Truth",
    )

    # Run CoTracker3 (~2 min on CPU)
    tracker = CoTracker3Online()
    with torch.no_grad():
        cotracker_pred_tracks, cotracker_pred_vis = tracker.forward(rgbs, query_points)
    cotracker_acc = compute_tracking_accuracy(
        pred_tracks=cotracker_pred_tracks,
        gt_tracks=gt_tracks,
        gt_vis=gt_vis,
        query_points=query_points,
        distance_threshold_px=distance_threshold_px,
    )
    viz_cotracker, _ = visualizer.visualize(
        video=rgbs[None],
        tracks=cotracker_pred_tracks[None],
        visibility=cotracker_pred_vis[None],
        query_frame=query_points_t[None],
        filename="cotracker",
        title=f"CoTracker3 (acc @ {distance_threshold_px}px: {cotracker_acc * 100:.2f}%)",
    )
    print(f"CoTracker3 Online accuracy @ {distance_threshold_px}px: {cotracker_acc * 100:.2f}%")
    # should print 92.94%

    # Run RAFT (~10 min on my CPU)
    raft_tracker = RaftPointTracker(checkpoint_path="checkpoints/raft-things.pth")
    with torch.no_grad():
        raft_pred_tracks, _ = raft_tracker.forward(rgbs, query_points)
    raft_acc = compute_tracking_accuracy(
        pred_tracks=raft_pred_tracks,
        gt_tracks=gt_tracks,
        gt_vis=gt_vis,
        query_points=query_points,
        distance_threshold_px=distance_threshold_px,
    )
    viz_raft, _ = visualizer.visualize(
        video=rgbs[None],
        tracks=raft_pred_tracks[None],
        visibility=None,
        query_frame=query_points_t[None],
        filename="raft",
        title=f"RAFT (acc @ {distance_threshold_px}px: {raft_acc * 100:.2f}%)",
    )
    print(f"RAFT accuracy @ {distance_threshold_px}px: {raft_acc * 100:.2f}%")
    # should print 60.82% (or 40–70% depending on implementation details and random factors)

    # Create additional visualization
    rows = []
    rows.append([viz_gt, viz_cotracker, viz_raft])
    for track_idx in [0, 5, 15]:
        row = []
        for title_prefix, filename, tracks, visib in [
            ("Ground Truth", "ground_truth", gt_tracks, gt_vis),
            ("CoTracker3", "cotracker", cotracker_pred_tracks, cotracker_pred_vis),
            ("RAFT", "raft", raft_pred_tracks, None),
        ]:
            acc = compute_tracking_accuracy(
                pred_tracks=tracks[:, track_idx:track_idx + 1],
                gt_tracks=gt_tracks[:, track_idx:track_idx + 1],
                gt_vis=gt_vis[:, track_idx:track_idx + 1],
                query_points=query_points[track_idx:track_idx + 1],
                distance_threshold_px=distance_threshold_px,
            )
            viz, _ = visualizer.visualize(
                video=rgbs[None],
                tracks=tracks[None, :, track_idx:track_idx + 1],
                visibility=None if visib is None else visib[None, :, track_idx:track_idx + 1],
                query_frame=query_points_t[track_idx:track_idx + 1][None],
                filename=f"track{track_idx}_{filename}",
                title=f"{title_prefix} - Track {track_idx} (acc @ {distance_threshold_px}px: {acc * 100:.2f}%)",
                save_video=False,
            )
            row.append(viz)
        rows.append(row)
    viz_final = torch.cat([torch.cat(r, dim=-1) for r in rows], dim=-2)
    visualizer.save_video(viz_final, visualizer.save_dir, "comparison", fps=visualizer.fps)
    print("Done. Visualizations saved to:", logs_dir)


if __name__ == "__main__":
    main()
