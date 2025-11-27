# Adapted from: https://github.com/ethz-vlg/mvtracker/blob/ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072/mvtracker/utils/visualizer_mp4.py

import logging
import os
import threading
from typing import Tuple

import cv2
import flow_vis
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib import cm
from moviepy.editor import ImageSequenceClip


def read_video_from_path(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {path}")

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frames.append(np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            break
    cap.release()

    return np.stack(frames)


class Visualizer:
    def __init__(
            self,
            save_dir: str = "./results",
            grayscale: bool = False,
            pad_value: int = 0,
            fps: int = 10,
            mode: str = "rainbow",  # 'cool', 'optical_flow'
            linewidth: int = 2,
            show_first_frame: int = 10,
            tracks_leave_trace: int = 0,  # -1 for infinite
            tracks_use_alpha: bool = False,
            print_debug_info: bool = False,
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.tracks_use_alpha = tracks_use_alpha
        self.print_debug_info = print_debug_info
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(
            self,
            video: torch.Tensor,  # (B,T,C,H,W)
            tracks: torch.Tensor,  # (B,T,N,2)
            visibility: torch.Tensor = None,  # (B, T, N) bool
            gt_tracks: torch.Tensor = None,  # (B,T,N,2)
            segm_mask: torch.Tensor = None,  # (B,1,H,W)
            filename: str = "video",
            writer=None,  # tensorboard Summary Writer, used for visualization during training
            step: int = 0,
            query_frame: torch.Tensor = None,  # (B,N)
            save_video: bool = True,
            compensate_for_camera_motion: bool = False,
            rigid_part=None,
            video_depth=None,  # (B,T,C,H,W)
            vector_colors=None,
            title: str = None,
    ):
        batch_size, num_frames, _, height, width = video.shape
        num_points = tracks.shape[-2]
        num_dims = tracks.shape[-1]

        assert video.shape == (batch_size, num_frames, 3, height, width)
        assert tracks.shape == (batch_size, num_frames, num_points, num_dims)
        if visibility is not None:
            assert visibility.shape == (batch_size, num_frames, num_points)
        if gt_tracks is not None:
            assert gt_tracks.shape == (batch_size, num_frames, num_points, num_dims)
        if query_frame is not None:
            assert query_frame.shape == (batch_size, num_points)

        if compensate_for_camera_motion:
            assert segm_mask is not None

        if segm_mask is not None:
            assert (query_frame == 0).all().item()
            coords = tracks[0, 0].round().long()
            segm_mask = segm_mask[0, 0][coords[:, 1], coords[:, 0]].long()

        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )
        if video_depth is not None:
            video_depth = video_depth.squeeze(2)
            video_depth = video_depth.cpu().numpy()
            highest_depth_value = max(video_depth.max(), 100)
            video_depth = plt.cm.Spectral(video_depth / highest_depth_value) * 255
            video_depth = video_depth[..., :3]
            video_depth = video_depth.astype(np.uint8)
            video_depth = torch.from_numpy(video_depth)
            video_depth = video_depth.permute(0, 1, 4, 2, 3)
            video_depth = F.pad(
                video_depth,
                (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
                "constant",
                255,
            )

        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        res_video, vector_colors = self.draw_tracks_on_video(
            video=video,
            tracks=tracks[..., :2],
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            rigid_part=rigid_part,
            vector_colors=vector_colors,
        )
        if video_depth is not None:
            res_video_depth, _ = self.draw_tracks_on_video(
                video=video_depth,
                tracks=tracks[..., :2],
                visibility=visibility,
                segm_mask=segm_mask,
                gt_tracks=gt_tracks,
                query_frame=query_frame,
                compensate_for_camera_motion=compensate_for_camera_motion,
                vector_colors=vector_colors,
            )
            res_video = torch.cat([res_video, res_video_depth], dim=4)  # B, T, 3, H, [W]

        if title is not None:
            B, T, C, H, W = res_video.shape
            title_height = 40  # px
            strip = np.full((T, title_height, W, 3), 0, dtype=np.uint8)
            for t in range(T):
                cv2.putText(
                    strip[t],
                    title,
                    (10, title_height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            strip_torch = torch.from_numpy(strip).permute(0, 3, 1, 2).unsqueeze(0)
            res_video = torch.cat([strip_torch, res_video], dim=3)

        if save_video:
            # self.save_video(res_video, filename=filename, writer=writer, step=step)
            thread = threading.Thread(
                target=Visualizer.save_video,
                args=(res_video, self.save_dir, filename, writer, self.fps, step)
            )
            thread.start()
        return res_video, vector_colors

    @staticmethod
    def save_video(video, save_dir, filename, writer=None, fps=12, step=0):
        if writer is not None:
            writer.add_video(f"{filename}", video.to(torch.uint8), global_step=step, fps=fps)
            writer.flush()
            logging.info(f"Video {filename} saved to tensorboard")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
            clip = ImageSequenceClip(wide_list, fps=fps)

            # Write the video file
            save_path = os.path.join(save_dir, f"{filename}.mp4")
            clip.write_videofile(save_path, codec="libx264", fps=fps, logger=None)

            logging.info(f"Video saved to {save_path}")

    def draw_tracks_on_video(
            self,
            video: torch.Tensor,
            tracks: torch.Tensor,
            visibility: torch.Tensor = None,
            segm_mask: torch.Tensor = None,
            gt_tracks=None,
            query_frame: torch.Tensor = None,
            compensate_for_camera_motion=False,
            vector_colors=None,
            rigid_part=None,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2
        if query_frame is not None:
            query_frame = query_frame[0].long().detach().cpu().numpy()  # N
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())

        if vector_colors is None:
            vector_colors = np.zeros((T, N, 3))
            if self.mode == "optical_flow":
                vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame, torch.arange(N)][None])
            elif segm_mask is None:
                if self.mode == "rainbow":
                    # y_min, y_max = (
                    #     tracks[query_frame, :, 1].min(),
                    #     tracks[query_frame, :, 1].max(),
                    # )
                    y_min, y_max = 0, H
                    norm = plt.Normalize(y_min, y_max)
                    for n in range(N):
                        color = self.color_map(norm(tracks[query_frame[n], n, 1]))
                        color = np.array(color[:3])[None] * 255
                        vector_colors[:, n] = np.repeat(color, T, axis=0)
                else:
                    # color changes with time
                    for t in range(T):
                        color = np.array(self.color_map(t / T)[:3])[None] * 255
                        vector_colors[t] = np.repeat(color, N, axis=0)
            else:
                if self.mode == "rainbow":
                    vector_colors[:, segm_mask <= 0, :] = 255

                    # y_min, y_max = (
                    #     tracks[0, segm_mask > 0, 1].min(),
                    #     tracks[0, segm_mask > 0, 1].max(),
                    # )
                    y_min, y_max = 0, H
                    norm = plt.Normalize(y_min, y_max)
                    for n in range(N):
                        if segm_mask[n] > 0:
                            color = self.color_map(norm(tracks[0, n, 1]))
                            color = np.array(color[:3])[None] * 255
                            vector_colors[:, n] = np.repeat(color, T, axis=0)

                else:
                    # color changes with segm class
                    segm_mask = segm_mask.cpu()
                    color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                    color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                    color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                    vector_colors = np.repeat(color[None], T, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind: t + 1]
                curr_colors = vector_colors[first_ind: t + 1]
                if compensate_for_camera_motion:
                    diff = (
                                   tracks[first_ind: t + 1, segm_mask <= 0]
                                   - tracks[t: t + 1, segm_mask <= 0]
                           ).mean(1)[:, None]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                    query_frame - first_ind,
                    use_alpha=self.tracks_use_alpha,
                )
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(
                        res_video[t], gt_tracks[first_ind: t + 1]
                    )

        # Add frame number
        if self.print_debug_info:
            for t in range(T):
                min_x = tracks[t].min(0)[0]
                min_y = tracks[t].min(0)[1]
                min_xy = f"{min_x:6.1f}, {min_y:6.1f}"

                median_x = np.median(tracks[t], axis=0)[0]
                median_y = np.median(tracks[t], axis=0)[1]
                median_xy = f"{median_x:6.1f}, {median_y:6.1f}"

                max_x = tracks[t].max(0)[0]
                max_y = tracks[t].max(0)[1]
                max_xy = f"{max_x:6.1f}, {max_y:6.1f}"

                text = (
                    f"Frame {t}"
                    f"\nH,W={H},{W}"
                    f"\nT,N={T},{N}"
                    f"\nmin_xy    = {min_xy} "
                    f"\nmedian_xy = {median_xy} "
                    f"\nmax_xy    = {max_xy} "
                )
                res_video[t] = put_debug_text_onto_image(res_video[t], text)

        if rigid_part is not None:
            cls_label = torch.unique(rigid_part)
            cls_num = len(torch.unique(rigid_part))
            # visualize the clustering results
            cmap = plt.get_cmap('jet')  # get the color mapping
            colors = cmap(np.linspace(0, 1, cls_num))
            colors = (colors[:, :3] * 255)
            color_map = {label.item(): color for label, color in zip(cls_label, colors)}

        #  draw points
        for t in range(T):
            for i in range(N):
                if query_frame is not None and query_frame[i] > t:
                    continue

                coord = (tracks[t, i, 0], tracks[t, i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]

                # Check for NaN or Inf in coordinates
                if np.isnan(coord).any() or np.isinf(coord).any():
                    logging.info(f"Warning: Skipping track {i} at t={t} due to NaN or Inf coord={coord}.")
                    continue  # Skip plotting this point

                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (
                            compensate_for_camera_motion and segm_mask[i] > 0
                    ):
                        if rigid_part is not None:
                            color = color_map[rigid_part.squeeze()[i].item()]
                            cv2.circle(
                                res_video[t],
                                coord,
                                int(self.linewidth * 2),
                                color.tolist(),
                                thickness=-1 if visibile else 2 - 1,
                            )
                        else:
                            cv2.circle(
                                res_video[t],
                                coord,
                                int(self.linewidth * 2),
                                vector_colors[t, i].tolist(),
                                thickness=-1 if visibile else 2 - 1,
                            )

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte(), vector_colors

    def _draw_pred_tracks(
            self,
            rgb: np.ndarray,  # H x W x 3
            tracks: np.ndarray,  # shape: [T, N, 2]
            vector_colors: np.ndarray,  # shape: [T, N, 3]
            query_frame: np.ndarray,  # shape: [N], each entry = birth frame for track i
            use_alpha: bool = False,
    ) -> np.ndarray:
        """
        Draws trajectory lines from frame s to s+1, but only if s >= query_frame[i].
        That is, no lines are drawn before the track 'appears' at query_frame[i].
        """
        T, N, _ = tracks.shape

        for s in range(T - 1):
            # We'll blend older lines more lightly (alpha) if desired:
            original_rgb = rgb.copy()
            if use_alpha:
                alpha = (s / T) ** 2  # or pick some function of s, T
            else:
                alpha = 1

            for i in range(N):
                # If the query/birth frame for track i is after s, skip drawing
                if query_frame is not None and s < query_frame[i]:
                    continue

                pt_s = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                pt_sp1 = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))

                # Skip if the points are 0 or invalid
                if pt_s[0] == 0 and pt_s[1] == 0:
                    continue
                if pt_sp1[0] == 0 and pt_sp1[1] == 0:
                    continue

                color = vector_colors[s, i].tolist()
                cv2.line(rgb, pt_s, pt_sp1, color, self.linewidth, cv2.LINE_AA)

            # Optionally alpha-blend older lines if you want them to fade out:
            rgb = cv2.addWeighted(rgb, alpha, original_rgb, 1 - alpha, 0)

        return rgb

    def _draw_gt_tracks(
            self,
            rgb: np.ndarray,  # H x W x 3,
            gt_tracks: np.ndarray,  # T x 2
    ):
        T, N, _ = gt_tracks.shape
        color = np.array((211.0, 0.0, 0.0))

        for t in range(T):
            for i in range(N):
                gt_tracks = gt_tracks[t][i]
                #  draw a red cross
                if gt_tracks[0] > 0 and gt_tracks[1] > 0:
                    length = self.linewidth * 3
                    coord_y = (int(gt_tracks[0]) + length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) - length, int(gt_tracks[1]) - length)
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                        cv2.LINE_AA,
                    )
                    coord_y = (int(gt_tracks[0]) - length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) + length, int(gt_tracks[1]) - length)
                    cv2.line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                        cv2.LINE_AA,
                    )
        return rgb


def put_debug_text_onto_image(img: np.ndarray, text: str, font_scale: float = 0.5, left: int = 5, top: int = 20,
                              font_thickness: int = 1, text_color_bg: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Overlay debug text on the provided image.

    Parameters
    ----------
    img : np.ndarray
        A 3D numpy array representing the input image. The image is expected to have three color channels.
    text : str
        The debug text to overlay on the image. The text can include newline characters ('\n') to create multi-line text.
    font_scale : float, default 0.5
        The scale factor that is multiplied by the font-specific base size.
    left : int, default 5
        The left-most coordinate where the text is to be put.
    top : int, default 20
        The top-most coordinate where the text is to be put.
    font_thickness : int, default 1
        Thickness of the lines used to draw the text.
    text_color_bg : Tuple[int, int, int], default (0, 0, 0)
        The color of the text background in BGR format.

    Returns
    -------
    img : np.ndarray
        A 3D numpy array representing the image with the debug text overlaid.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    font_color = (255, 255, 255)

    # Write each line of text in a new row
    (_, label_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    if text_color_bg is not None:
        for i, line in enumerate(text.split('\n')):
            (line_width, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            top_i = top + i * label_height
            cv2.rectangle(img, (left, top_i - label_height), (left + line_width, top_i), text_color_bg, -1)
    for i, line in enumerate(text.split('\n')):
        top_i = top + i * label_height
        cv2.putText(img, line, (left, top_i), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness)

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return img

