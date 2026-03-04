from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as Plt_Path
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.visualization.vis_utils import proj_lidar_bbox3d_to_img

from tools.detection3d.visualize_bev import OBJECT_PALETTE


@dataclass(frozen=True)
class DecodedBboxes:
    """
    Dataclass to save decoded bounding boxes from a 3d perception model and their metadata.
    :param lidar_bboxes: Decoded bboxes in lidar.
    :param lidar_pointclouds: Raw lidar pointclouds.
    :param scores: Scores for each bbox.
    :param labels: Labels for each bbox.
    :param class_name: Available class names for the outputs.
    :param img_paths: Available image paths for the outputs.
    :param lidar2cams [<4, 4>]: Intrinsic and extrinsic from lidar to cameras.
    :param cam2imgs [<3, 3>]: Intrinsic and extrinsic from cameras to images.
    """

    lidar_bboxes: LiDARInstance3DBoxes
    lidar_pointclouds: npt.NDArray[np.float64]
    scores: torch.tensor
    labels: torch.tensor
    class_names: List[str]
    img_paths: List[str]
    lidar2cams: List[npt.NDArray[np.float64]]
    cam2imgs: List[npt.NDArray[np.float64]]

    def project_lidar_bboxex_to_img(self, lidar2img: np.ndarray) -> npt.NDArray[np.float64]:
        """
        Project Bboxes in lidar view to an image view.
        :param lidar2img <4, 4>: intrinsic and extrinsic from lidar to an image.
        :return <N, 8, 2> (Number of bboxes, 8 corners, x and y coordinates).
        Projected bboxes in an image.
        """
        return proj_lidar_bbox3d_to_img(
            self.lidar_bboxes,
            input_meta={"lidar2img": lidar2img},
        )

    def compute_lidar_to_imgs(self) -> List[npt.NDArray[np.float64]]:
        """
        Compute Extrinsic and intrinsic from lidar to images.
        :return List of <4, 4> for extrinsic and intrinsic from a lidar to every images.
        """
        lidar2imgs = []
        for lidar2cam, cam2img in zip(self.lidar2cams, self.cam2imgs):
            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam2img).astype(np.float32)
            lidar2cam_array = np.asarray(lidar2cam, dtype=np.float32)
            lidar2imgs.append(cam2img_array @ lidar2cam_array)
        return lidar2imgs

    def visualize_bboxes_to_lidar(
        self,
        fig: Figure,
        grid_spec: GridSpec,
        xlim: Tuple[int, int],
        ylim: Tuple[int, int],
        radius: float = 0.1,
        thickness: float = 1 / 3,
        line_styles: str = "-",
        draw_front=True,
    ) -> None:
        """
        Visualize bboxes in LiDAR.
        :param ax: Axes to visualize bboxes in lidar.
        :param fpath: Path to save the visualization.
        :param xlim: Range in x-axis (-min, max).
        :param ylim: Range in y-axis (-min, max).
        :param draw_front: Set True to draw a line to indicate direction of bboxes.
        """
        ax = fig.add_subplot(grid_spec[-1, :], facecolor="white")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")

        ax.scatter(
            self.lidar_pointclouds[:, 0],
            self.lidar_pointclouds[:, 1],
            s=radius,
            c="black",
            marker=".",
            linewidths=0,
            edgecolors="none",
        )

        if self.lidar_bboxes is not None and len(self.lidar_bboxes) > 0:
            lines_verts_idx = [0, 3, 7, 4, 0]
            coords = self.lidar_bboxes.corners[:, lines_verts_idx, :2]
            codes = [Plt_Path.LINETO] * coords.shape[1]
            codes[0] = Plt_Path.MOVETO
            pathpatches = []
            edge_color_norms = []
            center_bottom_patches = []
            for index in range(coords.shape[0]):
                verts = coords[index]
                pth = Plt_Path(verts, codes)
                pathpatches.append(PathPatch(pth))

                label = self.labels[index]
                name = self.class_names[label]
                edge_color_rgb = np.array(OBJECT_PALETTE[name], dtype=np.float32) / 255
                edge_color_norms.append(np.concatenate([edge_color_rgb, np.array([0.5], dtype=np.float32)]))
                if draw_front:
                    # Draw line indicating the front
                    center_bottom_forward = torch.mean(
                        coords[index, 2:4, :2],
                        axis=0,
                        keepdim=True,
                    )
                    center_bottom = torch.mean(
                        coords[index, [0, 1, 2, 3], :2],
                        axis=0,
                        keepdim=True,
                    )
                    center_bottom_pth = Plt_Path(
                        torch.concat([center_bottom, center_bottom_forward], axis=0),
                        codes=codes[0:2],
                    )
                    center_bottom_patches.append(PathPatch(center_bottom_pth))

            p = PatchCollection(
                pathpatches,
                facecolors="none",
                edgecolors=edge_color_norms,
                linewidths=thickness,
                linestyles=line_styles,
            )
            ax.add_collection(p)

            if len(center_bottom_patches):
                line_collections = PatchCollection(
                    center_bottom_patches,
                    facecolors="none",
                    edgecolors=edge_color_norms,
                    linewidths=thickness,
                    linestyles=line_styles,
                )
                ax.add_collection(line_collections)

        ax.set_title("LiDAR")

    def visualize_bboxes_to_image(
        self,
        ax: plt.axes,
        img_path: str,
        lidar2img: npt.NDArray[np.float64],
        img_title_index: int = -1,
        alpha: float = 0.8,
        line_widths: float = 2 / 3,
        line_styles: str = "-",
    ) -> None:
        """
        Visualize bboxes from LiDAR to an image.
        This function is modified from mmdet3d.visualization.local_visualizer.draw_proj_bboxes_3d.
        :param ax: Matplotlib axis.
        :param img_path: Full image path.
        :param lidar2img <4, 4>: intrinsic and extrinsic from lidar to an image.
        :param img_title_index: Index to indicate an image name from img_path.
        :param alpha: Transparency of polygons.
        :param line_width: Thickness of bboxes' edges.
        :param line_styles: Style of bboxes's edges.
        :return np.float64 <N, 8, 2> (Number of bboxes, 8 corners, x and y coordinates).
        """
        # Draw the image to axis
        img = plt.imread(img_path)
        ax.imshow(img, interpolation="nearest")

        # Metadata about image
        h, w, _ = img.shape
        img_size = (w, h)
        path_parts = Path(img_path).parts
        camera_name = next((part for part in reversed(path_parts) if part.upper().startswith("CAM")), "")
        if camera_name:
            ax_title = camera_name
        else:
            ax_title = img_path.split("/")[img_title_index]

        corners_2d = self.project_lidar_bboxex_to_img(lidar2img=lidar2img)
        edge_color_norms = []
        face_color_norms = []
        if img_size is not None:
            # Filter out the bbox where there's no points in the images.
            # This is for the visualization of multi-view image.
            valid_point_idx = (
                (corners_2d[..., 0] >= 0)
                & (corners_2d[..., 0] <= img_size[0])
                & (corners_2d[..., 1] >= 0)
                & (corners_2d[..., 1] <= img_size[1])
            )  # noqa: E501
            valid_bbox_idx = valid_point_idx.sum(axis=-1) >= 1
            valid_bbox_labels = self.labels[valid_bbox_idx]
            corners_2d = corners_2d[valid_bbox_idx]
            for label in valid_bbox_labels:
                name = self.class_names[label]
                edge_color_rgb = np.array(OBJECT_PALETTE[name], dtype=np.float32) / 255
                face_color_norms.append(edge_color_rgb)
                edge_color_norms.append(np.concatenate([edge_color_rgb, np.array([0.5], dtype=np.float32)]))

        lines_verts_idx = [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 5, 1, 2, 6]
        lines_verts = corners_2d[:, lines_verts_idx, :]
        front_polys = corners_2d[:, 4:, :]
        codes = [Plt_Path.LINETO] * lines_verts.shape[1]
        codes[0] = Plt_Path.MOVETO
        pathpatches = []
        for i in range(len(corners_2d)):
            verts = lines_verts[i]
            pth = Plt_Path(verts, codes)
            pathpatches.append(PathPatch(pth))

        p = PatchCollection(
            pathpatches,
            facecolors="none",
            edgecolors=edge_color_norms,
            linewidths=line_widths,
            linestyles=line_styles,
        )
        ax.add_collection(p)

        # Draw a mask on the front of project bboxes
        front_polys = [front_poly for front_poly in front_polys]
        face_colors = face_color_norms
        polygon_collection = PolyCollection(
            front_polys,
            alpha=alpha,
            facecolor=face_colors,
            linestyles=line_styles,
            edgecolors=edge_color_norms,
            linewidths=line_widths,
        )
        ax.add_collection(polygon_collection)

        # Setting the axis
        ax.set_title(ax_title)
        ax.set_axis_off()

    def visualize_bboxes_to_images(
        self,
        fig: Figure,
        grid_spec: GridSpec,
        spec_cols: int = 3,
        alpha: float = 0.8,
    ) -> None:
        """
        Visualize bboxes from lidar to ever image.
        :param fpath: Path to save the visualization.
        :param ax_cols: Number of cols in the visualization.
        :param fig_size: Figure size.
        :param alpha: Transparency of polygons.
        """
        lidar2imgs = self.compute_lidar_to_imgs()
        assert len(self.img_paths) == len(lidar2imgs)
        def _camera_sort_key(img_and_lidar2img: Tuple[str, npt.NDArray[np.float64]]) -> str:
            img_path = img_and_lidar2img[0]
            path_parts = Path(img_path).parts
            camera_name = next((part for part in reversed(path_parts) if part.upper().startswith("CAM")), "")
            return camera_name if camera_name else img_path

        sorted_img_and_lidar2imgs = sorted(
            zip(self.img_paths, lidar2imgs),
            key=_camera_sort_key,
        )
        selected_row = 0
        for index, (img_path, lidar2img) in enumerate(
            sorted_img_and_lidar2imgs
        ):
            selected_col = index % spec_cols
            selected_row = index // spec_cols
            ax = fig.add_subplot(grid_spec[selected_row, selected_col])
            if img_path is not None:
                self.visualize_bboxes_to_image(
                    ax=ax,
                    img_path=img_path,
                    lidar2img=np.asarray(lidar2img),
                    alpha=alpha,
                    img_title_index=-2,
                )

    def visualize_bboxes(
        self,
        fpath: str,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        spec_cols: int = 3,
        alpha: float = 0.8,
        fig_size: Tuple[int, int] = (15, 15),
    ) -> None:
        """
        Visualize bboxes to both imgs and lidar in BEV.
        :param fpath: Path to save the visualization.
        :param xlim: x-axis range for lidar.
        :param ylim: y-axis range for lidar.
        :param spec_cols: Number of columns for gridspec in the visualization.
        :param alpha: Transparency of polygons in images.
        :param fig_size: Figure size.
        """
        # Init axes
        # Get the number of rows
        image_rows = ceil(len(self.img_paths) / spec_cols)
        fig = plt.figure(figsize=fig_size)

        # Images + Lidar
        grid_spec = fig.add_gridspec(image_rows + 1, spec_cols)

        # Add subplots for images
        if len(self.img_paths):
            self.visualize_bboxes_to_images(
                fig=fig,
                grid_spec=grid_spec,
                spec_cols=spec_cols,
                alpha=alpha,
            )

        # Add subplot for lidar
        self.visualize_bboxes_to_lidar(
            fig=fig,
            grid_spec=grid_spec,
            xlim=xlim,
            ylim=ylim,
            draw_front=True,
        )

        plt.tight_layout()
        plt.savefig(
            fname=fpath,
            format="png",
            # dpi=15,
            bbox_inches="tight",
        )
        plt.close()

    def visualize_bboxes_lidar_only(
        self,
        fpath: str,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        fig_size: Tuple[int, int] = (8, 8),
    ) -> None:
        """
        Visualize bboxes only in lidar BEV.
        :param fpath: Path to save the visualization.
        :param xlim: x-axis range for lidar.
        :param ylim: y-axis range for lidar.
        :param fig_size: Figure size.
        """
        fig = plt.figure(figsize=fig_size)
        grid_spec = fig.add_gridspec(1, 1)
        self.visualize_bboxes_to_lidar(
            fig=fig,
            grid_spec=grid_spec,
            xlim=xlim,
            ylim=ylim,
            draw_front=True,
        )
        plt.tight_layout()
        plt.savefig(
            fname=fpath,
            format="png",
            dpi=1000,
            bbox_inches="tight",
        )
        plt.close()

    def visualize_bboxes_camera_only(
        self,
        fpath: str,
        spec_cols: int = 3,
        alpha: float = 0.8,
        fig_size: Tuple[float, float] = (15, 10),
        dpi: int = 400,
    ) -> None:
        """
        Visualize bboxes only on camera images.
        :param fpath: Path to save the visualization.
        :param spec_cols: Number of columns for gridspec in the visualization.
        :param alpha: Transparency of polygons in images.
        :param fig_size: Figure size.
        """
        if len(self.img_paths) == 0:
            return

        image_rows = ceil(len(self.img_paths) / spec_cols)
        first_valid_img_path = next((img_path for img_path in self.img_paths if img_path is not None), None)
        if first_valid_img_path is not None:
            first_img = plt.imread(first_valid_img_path)
            img_h, img_w = first_img.shape[:2]
            fig_size = (
                max((img_w * spec_cols) / dpi, 1.0),
                max((img_h * image_rows) / dpi, 1.0),
            )

        fig = plt.figure(figsize=fig_size, dpi=dpi)
        grid_spec = fig.add_gridspec(image_rows, spec_cols)
        self.visualize_bboxes_to_images(
            fig=fig,
            grid_spec=grid_spec,
            spec_cols=spec_cols,
            alpha=alpha,
        )
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.97, wspace=0.02, hspace=0.10)
        plt.savefig(
            fname=fpath,
            format="png",
            dpi=dpi,
        )
        plt.close()


@dataclass(frozen=True)
class BatchDecodedBboxes:
    """Dataclass to save a batch of decoded bboxes with their meta information."""

    scene_name: str
    lidar_filename: str
    decoded_bboxes: DecodedBboxes

    def visualize(
        self,
        vis_dir: Path,
        xlim: Tuple[int, int],
        ylim: Tuple[int, int],
    ) -> None:
        """ """
        scene_path = vis_dir / self.scene_name
        scene_path.mkdir(exist_ok=True, parents=True)
        bev_scene_path = scene_path / "bev"
        cam_scene_path = scene_path / "cam"
        bev_scene_path.mkdir(exist_ok=True, parents=True)
        cam_scene_path.mkdir(exist_ok=True, parents=True)

        lidar_stem = Path(self.lidar_filename).stem
        lidar_fpath = bev_scene_path / f"{lidar_stem}.png"
        camera_fpath = cam_scene_path / f"{lidar_stem}.png"

        # Visualize bboxes only in bev
        self.decoded_bboxes.visualize_bboxes_lidar_only(
            fpath=lidar_fpath,
            xlim=xlim,
            ylim=ylim,
        )

        # Visualize bboxes only in camera views
        self.decoded_bboxes.visualize_bboxes_camera_only(
            fpath=camera_fpath,
            alpha=0.5,
        )
