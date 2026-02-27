# -*- coding: utf-8 -*-
"""Online CVOGL/CVOGL-Seg data pipeline for evaluation.

This module reads original CVOGL pth files directly and performs:
1) SVI panorama -> monocular conversion online;
2) point coordinate projection to monocular image;
3) CVOGL-Seg mask loading/alignment;
4) unified sample outputs for v3 and baseline evaluators.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _pano_point_to_pinhole(
    u_pano: float,
    v_pano: float,
    w_pano: int,
    h_pano: int,
    w_cam: int,
    h_cam: int,
    yaw: float,
    pitch: float,
    fov_x: float,
    fov_y: float,
) -> Optional[Tuple[float, float]]:
    theta = (u_pano / float(w_pano)) * 2.0 * np.pi - np.pi
    phi = np.pi / 2.0 - (v_pano / float(h_pano)) * np.pi

    d_world = np.array(
        [
            np.cos(phi) * np.sin(theta),
            np.sin(phi),
            np.cos(phi) * np.cos(theta),
        ],
        dtype=np.float64,
    )

    r_yaw = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )
    r_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    r = r_yaw @ r_pitch

    d_cam = r.T @ d_world
    if d_cam[2] <= 0:
        return None
    if abs(np.arctan2(d_cam[0], d_cam[2])) > fov_x / 2.0:
        return None
    if abs(np.arctan2(d_cam[1], d_cam[2])) > fov_y / 2.0:
        return None

    fx = w_cam / (2.0 * np.tan(fov_x / 2.0))
    fy = h_cam / (2.0 * np.tan(fov_y / 2.0))
    cx, cy = w_cam / 2.0, h_cam / 2.0
    u = fx * d_cam[0] / d_cam[2] + cx
    v = cy - fy * d_cam[1] / d_cam[2]
    return float(u), float(v)


def _pano_to_mono_with_point(
    panorama_rgb: np.ndarray,
    point_xy: Tuple[float, float],
    output_size: int = 518,
    fov_deg: float = 90.0,
    heading_deg: float = 0.0,
    pitch_deg: float = 0.0,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]]]:
    h_pano, w_pano = panorama_rgb.shape[:2]
    u_prompt, v_prompt = float(point_xy[0]), float(point_xy[1])

    fov = np.deg2rad(float(fov_deg))
    yaw = np.deg2rad(float(heading_deg))
    pitch = np.deg2rad(float(pitch_deg))

    projected_xy = _pano_point_to_pinhole(
        u_pano=u_prompt,
        v_pano=v_prompt,
        w_pano=w_pano,
        h_pano=h_pano,
        w_cam=output_size,
        h_cam=output_size,
        yaw=yaw,
        pitch=pitch,
        fov_x=fov,
        fov_y=fov,
    )

    focal = output_size / (2.0 * np.tan(fov / 2.0))
    cx = cy = output_size / 2.0

    r_yaw = np.array(
        [
            [np.cos(yaw), 0.0, np.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-np.sin(yaw), 0.0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )
    r_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    r = r_yaw @ r_pitch

    x, y = np.meshgrid(np.arange(output_size), np.arange(output_size))
    x_cam = (x - cx) / focal
    y_cam = -(y - cy) / focal
    z_cam = np.ones_like(x_cam)
    dirs = np.stack([x_cam, y_cam, z_cam], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs_world = dirs @ r.T

    lon = np.arctan2(dirs_world[..., 0], dirs_world[..., 2])
    lat = np.arcsin(np.clip(dirs_world[..., 1], -1.0, 1.0))

    u = (lon + np.pi) / (2.0 * np.pi) * w_pano
    v = (np.pi / 2.0 - lat) / np.pi * h_pano
    u = np.mod(u, w_pano)
    v = np.clip(v, 0, h_pano - 1)

    mono = cv2.remap(
        panorama_rgb,
        u.astype(np.float32),
        v.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )
    return mono, projected_xy


def _decode_binary_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    out = (mask > 127).astype(np.uint8)
    th, tw = target_hw
    if out.shape[0] != th or out.shape[1] != tw:
        out = cv2.resize(out, (tw, th), interpolation=cv2.INTER_NEAREST)
        out = (out > 0).astype(np.uint8)
    return out


def _bbox_xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    sa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    sb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return float(inter / (sa + sb - inter + 1e-16))


def _estimate_size_category(mask: np.ndarray) -> str:
    ratio = float(mask.astype(bool).sum()) / float(mask.shape[0] * mask.shape[1] + 1e-6)
    if ratio < 0.05:
        return "small"
    if ratio < 0.20:
        return "medium"
    return "large"


def _estimate_shape_category(mask: np.ndarray) -> str:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return "regular"
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    box_area = max(1, (x2 - x1) * (y2 - y1))
    area = int(mask.astype(bool).sum())
    extent = float(area) / float(box_area)
    return "regular" if extent >= 0.70 else "irregular"


@dataclass
class CVOGLRecord:
    dataset_name: str
    query_path: str
    sat_path: str
    rsimg_name: str
    click_xy: Tuple[float, float]
    bbox_xyxy: np.ndarray
    class_name: str


class _MaskIndex:
    def __init__(self, mask_dir: str):
        self.mask_dir = mask_dir
        self.prefix_map: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        if not os.path.isdir(mask_dir):
            return
        for fn in os.listdir(mask_dir):
            if "--bbox(" not in fn:
                continue
            if not fn.lower().endswith(".jpg"):
                continue
            prefix = fn.split("--bbox(", 1)[0]
            bbox_txt = fn.split("--bbox(", 1)[1].rsplit(")", 1)[0]
            parts = bbox_txt.split(",")
            if len(parts) != 4:
                continue
            try:
                box = np.array([float(p) for p in parts], dtype=np.float32)
            except Exception:
                continue
            self.prefix_map.setdefault(prefix, []).append((os.path.join(mask_dir, fn), box))

    def find(self, rsimg_name: str, target_bbox: np.ndarray) -> Optional[str]:
        prefix = os.path.splitext(rsimg_name)[0]
        items = self.prefix_map.get(prefix, [])
        if not items:
            return None

        best_path = None
        best_iou = -1.0
        for path, box in items:
            iou = _bbox_xyxy_iou(target_bbox, box)
            if iou > best_iou:
                best_iou = iou
                best_path = path
        return best_path


class CVOGLSegOnlineDataset(Dataset):
    def __init__(
        self,
        cvogl_root: str,
        cvoglseg_root: str,
        split_name: str = "test",
        img_size: int = 518,
        fov_deg: float = 90.0,
        svi_headings: Sequence[float] = (-135.0, -45.0, 45.0, 135.0),
    ):
        self.cvogl_root = cvogl_root
        self.cvoglseg_root = cvoglseg_root
        self.split_name = split_name
        self.img_size = int(img_size)
        self.fov_deg = float(fov_deg)
        self.svi_headings = list(svi_headings)

        self.records: List[CVOGLRecord] = []
        self._build_records()

        self.mask_index_svi = _MaskIndex(
            os.path.join(cvoglseg_root, "CVOGL_SVI", "mask-satellite")
        )
        self.mask_index_drone = _MaskIndex(
            os.path.join(cvoglseg_root, "CVOGL_DroneAerial", "mask-satellite")
        )

    def _build_records(self) -> None:
        cfgs = [
            ("CVOGL_SVI", "ground"),
            ("CVOGL_DroneAerial", "drone"),
        ]
        for dataset_name, _task in cfgs:
            pth_path = os.path.join(
                self.cvogl_root,
                dataset_name,
                f"{dataset_name}_{self.split_name}.pth",
            )
            if not os.path.isfile(pth_path):
                continue
            data_list = torch.load(pth_path, map_location=torch.device("cpu"))
            query_dir = os.path.join(self.cvogl_root, dataset_name, "query")
            sat_dir = os.path.join(self.cvogl_root, dataset_name, "satellite")

            for item in data_list:
                if not isinstance(item, (list, tuple)) or len(item) < 8:
                    continue
                query_name = str(item[1])
                sat_name = str(item[2])
                click_xy = item[4]
                bbox = item[5]
                class_name = str(item[7])
                if click_xy is None or bbox is None:
                    continue

                bbox_arr = np.array(bbox, dtype=np.float32).reshape(-1)[:4]
                self.records.append(
                    CVOGLRecord(
                        dataset_name=dataset_name,
                        query_path=os.path.join(query_dir, query_name),
                        sat_path=os.path.join(sat_dir, sat_name),
                        rsimg_name=sat_name,
                        click_xy=(float(click_xy[0]), float(click_xy[1])),
                        bbox_xyxy=bbox_arr.copy(),
                        class_name=class_name,
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    def _load_rgb(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _svi_project(self, pano_rgb: np.ndarray, click_xy: Tuple[float, float]) -> Tuple[np.ndarray, Tuple[float, float]]:
        best_img = None
        best_xy = None
        for heading in self.svi_headings:
            mono_img, point_xy = _pano_to_mono_with_point(
                pano_rgb,
                click_xy,
                output_size=self.img_size,
                fov_deg=self.fov_deg,
                heading_deg=float(heading),
                pitch_deg=0.0,
            )
            if point_xy is not None:
                best_img = mono_img
                best_xy = point_xy
                break
        if best_img is None:
            theta = (click_xy[0] / float(pano_rgb.shape[1])) * 360.0 - 180.0
            mono_img, point_xy = _pano_to_mono_with_point(
                pano_rgb,
                click_xy,
                output_size=self.img_size,
                fov_deg=self.fov_deg,
                heading_deg=float(theta),
                pitch_deg=0.0,
            )
            best_img = mono_img
            if point_xy is None:
                point_xy = (self.img_size * 0.5, self.img_size * 0.5)
            best_xy = point_xy
        return best_img, (float(best_xy[0]), float(best_xy[1]))

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        task_type = "ground" if rec.dataset_name == "CVOGL_SVI" else "drone"

        q = self._load_rgb(rec.query_path)
        s = self._load_rgb(rec.sat_path)
        h_s, w_s = s.shape[:2]

        if rec.dataset_name == "CVOGL_SVI":
            query_img, click_xy = self._svi_project(q, rec.click_xy)
        else:
            query_img = cv2.resize(q, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            sx = self.img_size / float(max(1, q.shape[1]))
            sy = self.img_size / float(max(1, q.shape[0]))
            click_xy = (float(rec.click_xy[0] * sx), float(rec.click_xy[1] * sy))

        # CVOGL bbox is anchored to 1024 by default in this dataset.
        bbox = rec.bbox_xyxy.astype(np.float32).copy()
        sx_s = float(w_s) / 1024.0
        sy_s = float(h_s) / 1024.0
        bbox[0::2] *= sx_s
        bbox[1::2] *= sy_s

        mask_idx = self.mask_index_svi if rec.dataset_name == "CVOGL_SVI" else self.mask_index_drone
        mask_path = mask_idx.find(rec.rsimg_name, bbox)
        if mask_path and os.path.isfile(mask_path):
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_raw is None:
                gt_mask = np.zeros((h_s, w_s), dtype=np.uint8)
            else:
                gt_mask = _decode_binary_mask(mask_raw, (h_s, w_s))
        else:
            gt_mask = np.zeros((h_s, w_s), dtype=np.uint8)
            x1, y1, x2, y2 = bbox.astype(np.int32).tolist()
            x1 = max(0, min(x1, w_s - 1))
            y1 = max(0, min(y1, h_s - 1))
            x2 = max(0, min(x2, w_s))
            y2 = max(0, min(y2, h_s))
            if x2 > x1 and y2 > y1:
                gt_mask[y1:y2, x1:x2] = 1

        sat_img = cv2.resize(s, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # gt_mask is already binary (0/1), so just resize without re-thresholding at 127
        if gt_mask.shape[0] != self.img_size or gt_mask.shape[1] != self.img_size:
            gt_mask = cv2.resize(gt_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            gt_mask = (gt_mask > 0).astype(np.uint8)

        scale_x = self.img_size / float(max(1, w_s))
        scale_y = self.img_size / float(max(1, h_s))
        bbox[0::2] *= scale_x
        bbox[1::2] *= scale_y
        bbox[0::2] = np.clip(bbox[0::2], 0, self.img_size - 1)
        bbox[1::2] = np.clip(bbox[1::2], 0, self.img_size - 1)

        click_xy = (
            float(np.clip(click_xy[0], 0, self.img_size - 1)),
            float(np.clip(click_xy[1], 0, self.img_size - 1)),
        )

        size_category = _estimate_size_category(gt_mask)
        shape_category = _estimate_shape_category(gt_mask)

        return {
            "query_img": query_img.astype(np.uint8),
            "sat_img": sat_img.astype(np.uint8),
            "point_xy": np.array(click_xy, dtype=np.float32),
            "gt_bbox_xyxy": bbox.astype(np.float32),
            "gt_mask": gt_mask.astype(np.uint8),
            "task_type": task_type,
            "size_category": size_category,
            "shape_category": shape_category,
            "dataset_name": rec.dataset_name,
            "index": int(idx),
            "class_name": rec.class_name,
            "query_path": rec.query_path,
            "sat_path": rec.sat_path,
        }

