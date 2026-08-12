# Domino chain-reaction game for aim_fsm, combined with the kickpickrefine
# knock/pickup/place pipeline using 4 explicit geometric case routines.
#
# Run with:  runfsm("Game")

from __future__ import annotations

import math
import os
import time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0
from ultralytics import YOLO

from aim_fsm import *
from aim_fsm.worldmap import DominoObj
from aim_fsm.domino import DominoWorldDetector, normalize_axis_angle

# ----------------------------------------------------------------------
# Tunable constants -- sweep / world-map game logic
# ----------------------------------------------------------------------
SWEEP_STEP_MM = 50.0        # 5 cm per scan stop
SWEEP_LEG1_MM = 100.0       # 10 cm right
SWEEP_LEG2_MM = 200.0
SWEEP_LEG3_MM = 100.0       # 10 cm left

REPOSITION_LEFT_MM = 150.0     # 15 cm
REPOSITION_FORWARD_MM = 200.0  # 20 cm

SCAN_SETTLE_SECONDS = 0.3   # let camera/robot settle
SCAN_MAX_SECONDS = 3.0      # max scan stop duration
SCAN_POLL_SECONDS = 0.2

APPROACH_STANDOFF_MM = 50.0  # coarse approach standoff

KNOCK_SIDEWAYS_MM = 20.0     # 2 cm sideways motion for knock sequence
KNOCK_RETREAT_MM = 30.0      # 3 cm back off after knock to bring domino into FOV

POST_RETURN_FORWARD_MM = 150.0 # 15 cm forward move after returning to origin
POST_RETURN_RIGHT_MM = 100.0   # 10 cm right shift before passing turn to human

SWEEP_RIGHT_SIGN = 1.0
SNAP_TO_GRID_AXIS = True

# Scale factor to compensate for camera foreshortening/world map overestimation
WORLD_MAP_SCALE_FACTOR = 0.85

STANDING_WEIGHTS = "bestieee.pt"
STANDING_LABEL_WEIGHTS = "different.pt"
FALLEN_WEIGHTS = "fallen.pt"
FALLEN_LABEL_WEIGHTS = "fallenhalf.pt"

# ----------------------------------------------------------------------
# Tunable constants -- kickpickrefine pipeline
# ----------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWN_LENGTH = 4.8      # Real domino length in cm
FOCAL_LENGTH = 396.3    # Focal length
CAMERA_HFOV_DEG = 62.0  # Horizontal angle ref

SEG_MODEL_PATH = os.path.join(PROJECT_DIR, STANDING_WEIGHTS)
CLS_MODEL_PATH = os.path.join(PROJECT_DIR, STANDING_LABEL_WEIGHTS)
FALLEN_CLS_MODEL_PATH = os.path.join(PROJECT_DIR, FALLEN_LABEL_WEIGHTS)
FALLEN_MODEL_PRIMARY = os.path.join(PROJECT_DIR, FALLEN_WEIGHTS)
FALLEN_MODEL_FALLBACK = os.path.join(
    PROJECT_DIR, "runs_play_dominos", "play_dominos_seg_v1_img160", "weights", STANDING_WEIGHTS
)

NUM_CLASSES = 7
MIN_HALF_SIZE = 4
WHITE_THRESHOLD = 250
BORDER_MARGIN = 4

SETTLE_DELAY_SEC = 0.20
MODEL_SIZE = 320
PREDICT_CONF = 0.01
PREDICT_IOU = 0.45
DETECT_TRIES = 5
DETECT_RETRY_DELAY_SEC = 0.15

ALIGN_TOL_PX_NEAR = 4.0
ALIGN_TOL_PX_FAR = 10.0
NEAR_ALIGN_DISTANCE_CM = 18.0
MIN_TURN_DEG = 2.0
MAX_TURN_DEG = 10.0
MAX_TURN_STEPS = 6

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PYTORCH_TENSOR_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ==========================================
# Free helper functions
# ==========================================
def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def sign(value):
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def wrap_deg(deg):
    while deg > 180.0:
        deg -= 360.0
    while deg < -180.0:
        deg += 360.0
    return deg


def estimate_angle_and_distance_cm(center_x: float, image_w: float, major_length_px: float):
    focal_px_for_angle = (image_w * 0.5) / math.tan(math.radians(CAMERA_HFOV_DEG * 0.5))
    angle_deg = math.degrees(math.atan2(center_x - (image_w * 0.5), focal_px_for_angle))
    distance_cm = (KNOWN_LENGTH * FOCAL_LENGTH) / max(major_length_px, 1.0)
    return angle_deg, distance_cm


def estimate_divider_x(image_bgr: np.ndarray, x1: float, y1: float, x2: float, y2: float):
    h, w = image_bgr.shape[:2]
    ix1 = max(0, min(int(round(x1)), w - 1))
    iy1 = max(0, min(int(round(y1)), h - 1))
    ix2 = max(ix1 + 1, min(int(round(x2)), w))
    iy2 = max(iy1 + 1, min(int(round(y2)), h))

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    roi = image_bgr[iy1:iy2, ix1:ix2]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    rw = gray.shape[1]
    if rw < 12:
        return None

    c0 = int(0.30 * rw)
    c1 = int(0.70 * rw)
    if c1 <= c0 + 1:
        return None

    profile = gray.mean(axis=0)
    band = profile[c0:c1]
    local = int(np.argmin(band))
    divider_x_local = c0 + local

    darkest = float(band[local])
    median = float(np.median(band))
    if (median - darkest) < 8.0:
        return None

    return float(ix1 + divider_x_local)


def roboflow_fit_resize(img, size=MODEL_SIZE):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y, new_w, new_h


def map_mask_to_original(mask_reduced, pad_x, pad_y, new_w, new_h, orig_w, orig_h):
    binary_mask = (mask_reduced > 0.1).astype(np.uint8)
    cropped = binary_mask[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    if cropped.size == 0 or new_w <= 0 or new_h <= 0:
        return np.zeros((orig_h, orig_w), dtype=np.uint8)
    restored = cv2.resize(
        cropped,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return (restored * 255).astype(np.uint8)


def long_axis_from_rect(rect):
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w
        angle += 90.0

    theta = math.radians(angle)
    ux, uy = math.cos(theta), math.sin(theta)

    p1 = np.array([cx - 0.5 * w * ux, cy - 0.5 * w * uy], dtype=np.float32)
    p2 = np.array([cx + 0.5 * w * ux, cy + 0.5 * w * uy], dtype=np.float32)

    return p1, p2, angle, w


def domino_ground_contact_px(rect):
    box_pts = cv2.boxPoints(rect).astype(np.float32)
    sorted_by_y = sorted(box_pts, key=lambda pt: pt[1], reverse=True)
    p_base1, p_base2 = sorted_by_y[0], sorted_by_y[1]
    return float((p_base1[0] + p_base2[0]) / 2.0), float((p_base1[1] + p_base2[1]) / 2.0)


def project_pixel_to_base(robot, px, py, bbox_long_side_px=None, frame_width=640):
    try:
        hit = robot.kine.project_to_ground(float(px), float(py)).copy()
        return float(hit[0][0]), float(hit[1][0])
    except Exception as exc:
        print(f"[project_pixel_to_base] Kinematic projection failed ({exc}); using size-based fallback.")

    if bbox_long_side_px is not None and bbox_long_side_px > 0:
        CAMERA_BASE_OFFSET_CM = 3.2
        CORRECTED_FOCAL = FOCAL_LENGTH * 1.25
        direct_dist_cm = ((CORRECTED_FOCAL * KNOWN_LENGTH) / bbox_long_side_px) + CAMERA_BASE_OFFSET_CM

        center_offset_px = px - (frame_width / 2.0)
        angle_rad = math.atan2(center_offset_px, FOCAL_LENGTH)

        x_mm = direct_dist_cm * 10.0 * math.cos(angle_rad)
        y_mm = direct_dist_cm * 10.0 * math.sin(angle_rad)
        return x_mm, y_mm

    print("[project_pixel_to_base] No kinematic projection and no bbox size available; returning (0, 0).")
    return 0.0, 0.0


def build_fallen_candidates(game_fsm, image, result, pad_x, pad_y, new_w, new_h, scale, w, h):
    candidates = []
    mask_data = result.masks.data

    for i in range(len(mask_data)):
        mask_small = mask_data[i].detach().cpu().numpy()
        if np.count_nonzero(mask_small > 0.1) == 0:
            continue

        mask_orig = map_mask_to_original(mask_small, pad_x, pad_y, new_w, new_h, w, h)
        mask_area = float(cv2.countNonZero(mask_orig))
        contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            small_contours, _ = cv2.findContours((mask_small > 0.1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not small_contours:
                continue
            cnt_small = max(small_contours, key=cv2.contourArea)
            rect_small = cv2.minAreaRect(cnt_small)
            (cx_s, cy_s), (rw_s, rh_s), ang_s = rect_small
            cx = (cx_s - pad_x) / scale
            cy = (cy_s - pad_y) / scale
            rect = ((cx, cy), (rw_s / scale, rh_s / scale), ang_s)
            mask_area = float(cv2.contourArea(cnt_small)) / (scale * scale)
        else:
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            (cx, cy), _, _ = rect

        p1, p2, orientation_deg, long_side_px = long_axis_from_rect(rect)

        left_pred = right_pred = None
        left_conf = right_conf = 0.0
        try:
            box_pts = cv2.boxPoints(rect).astype(np.float32)
            crop, _crop_bbox, _crop_major_len = game_fsm.rotate_crop(image, box_pts)
        except Exception:
            crop = None

        if crop is not None and crop.size > 0:
            left_raw, right_raw = game_fsm.split_halves(crop)
            if left_raw is not None and right_raw is not None:
                left_clean = game_fsm.remove_white_border(left_raw)
                right_clean = game_fsm.remove_white_border(right_raw)
                left_tensor = game_fsm.preprocess_half(left_clean)
                right_tensor = game_fsm.preprocess_half(right_clean)
                left_pred, left_conf = game_fsm.predict_half(left_tensor, game_fsm.fallen_classifier)
                right_pred, right_conf = game_fsm.predict_half(right_tensor, game_fsm.fallen_classifier)

        label_fwd = f"{left_pred}-{right_pred}" if left_pred is not None and right_pred is not None else None
        label_rev = f"{right_pred}-{left_pred}" if left_pred is not None and right_pred is not None else None

        candidates.append({
            "mask_area": mask_area,
            "cx": cx,
            "cy": cy,
            "rect": rect,
            "p1": p1,
            "p2": p2,
            "orientation_deg": orientation_deg,
            "long_side_px": long_side_px,
            "left_pred": left_pred,
            "right_pred": right_pred,
            "left_conf": left_conf,
            "right_conf": right_conf,
            "label_fwd": label_fwd,
            "label_rev": label_rev,
        })

    return candidates


class Game(StateMachineProgram):

    def __init__(self):
        super().__init__(
            launch_cam_viewer=True,
            launch_worldmap_viewer=True,
            launch_path_viewer=False,
            speech=False,
            aruco=False,
            domino=False,
            domino_labeling=False,
            force_annotation=True,
        )

        self.detector = DominoWorldDetector(
            conf_threshold=0.35,
            standing_weights=STANDING_WEIGHTS,
            fallen_weights=FALLEN_WEIGHTS,
            standing_label_weights=STANDING_LABEL_WEIGHTS,
            fallen_label_weights=FALLEN_LABEL_WEIGHTS,
            frame_skip=1,
        )

        self.match_target = None
        self.chain_open_target_obj = None
        self.target_labels: set[str] = set()
        self._pending_distance_mm = 0.0
        self._pending_sideways_mm = 0.0
        self.match_number: int | None = None

        self.a_near_pip: int | None = None
        self.a_far_pip: int | None = None
        self.b_open_pip: int | None = None
        self.b_inner_pip: int | None = None
        self.manipulation_case: int | None = None

        self.a_match_side: str | None = None
        self.a_needs_flip: bool = False

        self.return_pose = None
        self._return_distance_mm = 0.0

        self.fb_mm = 0.0
        self.lb_mm = 0.0
        self.fa_mm = 0.0
        self.la_mm = 0.0
        self.theta_a_deg = 0.0
        self.delta_f_mm = 0.0

        print("Initializing kickpickrefine visual align/knock/pickup pipeline...")

        if os.path.exists(SEG_MODEL_PATH):
            self.segmenter = YOLO(SEG_MODEL_PATH)
        else:
            print(f"Warning: Segmenter weights '{SEG_MODEL_PATH}' not found!")
            self.segmenter = None

        self.classifier = self._load_classifier(CLS_MODEL_PATH)

        if os.path.exists(FALLEN_CLS_MODEL_PATH):
            self.fallen_classifier = self._load_classifier(FALLEN_CLS_MODEL_PATH)
        else:
            print("Warning: Fallen classifier missing; using standing classifier.")
            self.fallen_classifier = self.classifier

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.target_data = None
        self.turn_steps = 0
        self.last_charge_distance_mm = 0.0
        self.actual_retreated_mm = 0.0

        self.measured_forward_cm = 0.0
        self.measured_lateral_cm = 0.0
        self.measured_dividing_angle_deg = 0.0

        self.post_turn_sideways_mm = 0.0
        self.post_turn_angle_rad = 0.0

        self.fallen_model_path = None
        if os.path.exists(FALLEN_MODEL_PRIMARY):
            self.fallen_model_path = FALLEN_MODEL_PRIMARY
        elif os.path.exists(FALLEN_MODEL_FALLBACK):
            self.fallen_model_path = FALLEN_MODEL_FALLBACK

        self.fallen_model = None
        if self.fallen_model_path is not None:
            print(f"Loading fallen segmentation weights: {self.fallen_model_path}")
            self.fallen_model = YOLO(self.fallen_model_path)
            self.fallen_model.eval()

        self.target = None
        self.debug_view = None

    def _sideways_right(self, mm: float) -> float:
        return SWEEP_RIGHT_SIGN * mm

    def _sideways_left(self, mm: float) -> float:
        return -SWEEP_RIGHT_SIGN * mm

    def _sweep_plan_mm(self) -> list[float]:
        steps: list[float] = []
        n1 = int(round(SWEEP_LEG1_MM / SWEEP_STEP_MM))
        n2 = int(round(SWEEP_LEG2_MM / SWEEP_STEP_MM))
        n3 = int(round(SWEEP_LEG3_MM / SWEEP_STEP_MM))
        steps += [self._sideways_right(SWEEP_STEP_MM)] * n1
        steps += [self._sideways_left(SWEEP_STEP_MM)] * n2
        steps += [self._sideways_right(SWEEP_STEP_MM)] * n3
        return steps

    def _place_domino_observation(self, obs, is_fallen: bool):
        world_map = self.robot.world_map

        if not obs.face_label:
            return None

        clean_label = obs.face_label.split(".")[0]

        with world_map._lock:
            standing_id = f"Domino-standing-{clean_label}"
            if is_fallen and standing_id in world_map.objects:
                del world_map.objects[standing_id]

        quad = np.array(obs.quad, dtype=np.float32) if getattr(obs, "quad", None) is not None else None
        if quad is None or len(quad) != 4:
            return None

        cx, _cy = obs.center_xy
        bottom_cy = float(np.max(quad[:, 1]))
        hit, objpos = world_map.project_image_point_to_world(cx, bottom_cy)
        if objpos is None:
            return None

        x_mm = float(objpos[0][0])
        y_mm = float(objpos[1][0])

        sorted_by_y = sorted(quad, key=lambda pt: pt[1], reverse=True)
        p_base1, p_base2 = sorted_by_y[0], sorted_by_y[1]
        if p_base1[0] > p_base2[0]:
            p_base1, p_base2 = p_base2, p_base1

        hit1, pos1 = world_map.project_image_point_to_world(p_base1[0], p_base1[1])
        hit2, pos2 = world_map.project_image_point_to_world(p_base2[0], p_base2[1])

        if pos1 is None or pos2 is None:
            local_yaw = 0.0
        else:
            dx_world = float(pos2[0][0] - pos1[0][0])
            dy_world = float(pos2[1][0] - pos1[1][0])
            local_yaw = math.atan2(dy_world, dx_world)

        raw_world_yaw = normalize_axis_angle(self.robot.pose.theta + local_yaw)

        if SNAP_TO_GRID_AXIS:
            pi_half = math.pi / 2.0
            world_yaw = round(raw_world_yaw / pi_half) * pi_half
            world_yaw = normalize_axis_angle(world_yaw)
        else:
            world_yaw = raw_world_yaw

        exact_obj_id = f"Domino-{'fallen' if is_fallen else 'standing'}-{clean_label}"

        with world_map._lock:
            if exact_obj_id in world_map.objects:
                domino_obj = world_map.objects[exact_obj_id]
                domino_obj.pose.x = x_mm
                domino_obj.pose.y = y_mm
                domino_obj.pose.theta = world_yaw
                domino_obj.is_fallen = is_fallen
            else:
                domino_obj = DominoObj(
                    id=exact_obj_id,
                    x=x_mm,
                    y=y_mm,
                    z=4.0 if is_fallen else 12.0,
                    theta=world_yaw,
                    face_label=clean_label,
                    face_confidence=getattr(obs, "face_confidence", None),
                    is_fallen=is_fallen,
                )
                world_map.objects[exact_obj_id] = domino_obj

            domino_obj.is_visible = True
            domino_obj.is_missing = False
            domino_obj.is_static = True
            domino_obj.confidence = 1.0
            domino_obj.last_updated = time.time()

            if hasattr(world_map, "dispatch_update"):
                world_map.dispatch_update()

        return domino_obj

    def _load_classifier(self, path):
        try:
            model = efficientnet_b0(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            return model
        except Exception as exc:
            print(f"ERROR loading classifier: {exc}")
            return None

    def rotate_crop(self, image, obb):
        points = obb.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(points)
        center, size, angle = rect
        width, height = size

        if width <= 0 or height <= 0:
            return None, None, 0.0

        major_length = max(width, height)

        if width < height:
            angle += 90
            width, height = height, width

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
        )

        x, y = int(center[0]), int(center[1])
        pad = 20

        x1 = max(0, x - int(width / 2) - pad)
        y1 = max(0, y - int(height / 2) - pad)
        x2 = min(image.shape[1], x + int(width / 2) + pad)
        y2 = min(image.shape[0], y + int(height / 2) + pad)

        crop = rotated[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None, None, 0.0

        return crop, (x1, y1, x2, y2), major_length

    def split_halves(self, image):
        try:
            if image is None or image.size == 0:
                return None, None
            h, w = image.shape[:2]
            mid = w // 2
            if mid < MIN_HALF_SIZE or (w - mid) < MIN_HALF_SIZE:
                return None, None
            return image[:, :mid].copy(), image[:, mid:].copy()
        except Exception:
            return None, None

    def remove_white_border(self, image, threshold=WHITE_THRESHOLD, margin=BORDER_MARGIN):
        try:
            if image is None or image.size == 0:
                return image
            keep_mask = np.any(image < threshold, axis=2)
            if not np.any(keep_mask):
                return image
            ys, xs = np.where(keep_mask)
            y0 = max(0, int(ys.min()) - margin)
            y1 = min(image.shape[0], int(ys.max()) + margin + 1)
            x0 = max(0, int(xs.min()) - margin)
            x1 = min(image.shape[1], int(xs.max()) + margin + 1)
            trimmed = image[y0:y1, x0:x1]
            return trimmed if (trimmed is not None and trimmed.size > 0) else image
        except Exception:
            return image

    def preprocess_half(self, half_bgr):
        try:
            if half_bgr is None or half_bgr.size == 0 or half_bgr.shape[0] < MIN_HALF_SIZE or half_bgr.shape[1] < MIN_HALF_SIZE:
                return None
            gray = cv2.cvtColor(half_bgr, cv2.COLOR_BGR2GRAY)
            filtered = cv2.bilateralFilter(gray, 5, 75, 75)

            if self.clahe is None:
                self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            equalized = self.clahe.apply(filtered)

            three_channel = np.repeat(equalized[:, :, np.newaxis], 3, axis=2)
            pil_image = Image.fromarray(three_channel)
            tensor = PYTORCH_TENSOR_TRANSFORMS(pil_image)
            return tensor.unsqueeze(0)
        except Exception:
            return None

    def predict_half(self, tensor, classifier=None):
        try:
            model = classifier if classifier is not None else self.classifier
            if model is None or tensor is None:
                return None, 0.0
            with torch.no_grad():
                tensor = tensor.to(DEVICE)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)
                pred_class = int(predicted.item())
                conf_value = float(confidence.item())
            if pred_class < 0 or pred_class >= NUM_CLASSES:
                return None, 0.0
            return pred_class, conf_value
        except Exception:
            return None, 0.0

    # ------------------------------------------------------------------
    # FSM Nodes
    # ------------------------------------------------------------------

    class ResetGame(StateNode):
        def start(self, event=None):
            super().start(event)
            self.parent.match_target = None
            self.parent.chain_open_target_obj = None
            self.parent.target_labels = set()
            self.parent._pending_distance_mm = 0.0
            self.parent._pending_sideways_mm = 0.0
            self.parent.match_number = None
            self.parent.a_near_pip = None
            self.parent.a_far_pip = None
            self.parent.b_open_pip = None
            self.parent.b_inner_pip = None
            self.parent.manipulation_case = None
            self.parent.a_match_side = None
            self.parent.a_needs_flip = False
            self.parent.return_pose = None
            self.parent._return_distance_mm = 0.0
            self.parent.fb_mm = 0.0
            self.parent.lb_mm = 0.0
            self.parent.fa_mm = 0.0
            self.parent.la_mm = 0.0
            self.parent.theta_a_deg = 0.0
            self.parent.delta_f_mm = 0.0
            self.post_completion()

    class SidewaysMove(ActionNode):
        def __init__(self, distance_mm: float = 0.0):
            super().__init__()
            self.distance_mm = distance_mm

        def start(self, event=None):
            super().start(event)
            self.robot.actuators["drive"].sideways(self, self.distance_mm, None)

    class ForwardMove(ActionNode):
        def __init__(self, distance_mm: float = 0.0):
            super().__init__()
            self.distance_mm = distance_mm

        def start(self, event=None):
            super().start(event)
            self.robot.actuators["drive"].forward(self, self.distance_mm, None)

    class ScanAndPlace(StateNode):
        def __init__(self, expected_fallen: bool = False):
            super().__init__()
            self.expected_fallen = expected_fallen

        def start(self, event=None):
            super().start(event)
            parent = self.parent
            detector = parent.detector

            time.sleep(SCAN_SETTLE_SECONDS)
            deadline = time.time() + max(0.0, SCAN_MAX_SECONDS - SCAN_SETTLE_SECONDS)

            placed = []
            while time.time() < deadline:
                image = self.robot.camera_image
                if image is None:
                    time.sleep(SCAN_POLL_SECONDS)
                    continue
                try:
                    observations = detector.detect(
                        image, frame_id=getattr(self.robot, "frame_count", None)
                    )
                except Exception:
                    observations = []

                relevant = [
                    o for o in observations
                    if bool(getattr(o, "is_fallen", False)) == self.expected_fallen
                ]
                if relevant:
                    for obs in relevant:
                        obj = parent._place_domino_observation(obs, self.expected_fallen)
                        if obj is not None:
                            placed.append(obj)
                    break
                time.sleep(SCAN_POLL_SECONDS)

            kind = "fallen" if self.expected_fallen else "standing"
            if placed:
                print(f"[Game] Processed {len(placed)} {kind} domino(s)")
            else:
                print(f"[Game] No {kind} dominoes seen at this stop.")
            self.post_completion()

    class ReportStanding(StateNode):
        def start(self, event=None):
            super().start(event)
            world_map = self.robot.world_map
            with world_map._lock:
                standing = [
                    o for o in world_map.objects.values()
                    if isinstance(o, DominoObj) and not o.is_fallen
                    and not getattr(o, "is_missing", False)
                ]
            print(f"[Game] Live World Map Standing Sweep Complete: {len(standing)} active dominoes.")
            for obj in standing:
                print(f"    {obj}")
            self.post_completion()

    class AnalyzeChain(StateNode):
        def start(self, event=None):
            super().start(event)
            world_map = self.robot.world_map

            with world_map._lock:
                fallen = [
                    o for o in world_map.objects.values()
                    if isinstance(o, DominoObj) and o.is_fallen and o.face_label
                    and not getattr(o, "is_missing", False)
                ]
                standing = [
                    o for o in world_map.objects.values()
                    if isinstance(o, DominoObj) and not o.is_fallen and o.face_label
                    and not getattr(o, "is_missing", False)
                ]

            if not fallen:
                print("[Game] No real fallen dominoes on Live World Map; passing turn to human.")
                self.parent.match_target = None
                self.parent.chain_open_target_obj = None
                self.parent.target_labels = set()
                self.post_failure()
                return

            counts = Counter()
            pairs = []
            for obj in fallen:
                try:
                    a_str, b_str = obj.face_label.split("-")
                    a, b = int(a_str), int(b_str)
                except (ValueError, AttributeError):
                    continue
                pairs.append((a, b))
                counts[a] += 1
                counts[b] += 1

            if len(pairs) == 1:
                ends = list(pairs[0])
            else:
                ends = sorted(v for v, c in counts.items() if c % 2 != 0)

            print(f"[Game] Live Fallen chain pairs ({len(pairs)}): {pairs}")
            print(f"[Game] Calculated chain open ends: {ends}")

            if not ends:
                self.parent.match_target = None
                self.parent.chain_open_target_obj = None
                self.parent.target_labels = set()
                self.post_failure()
                return

            chain_end_obj = fallen[0]
            for obj in fallen:
                try:
                    a, b = map(int, obj.face_label.split("-"))
                    if a in ends or b in ends:
                        chain_end_obj = obj
                        break
                except Exception:
                    pass

            self.parent.chain_open_target_obj = chain_end_obj

            matches = []
            for obj in standing:
                try:
                    a_str, b_str = obj.face_label.split("-")
                    a, b = int(a_str), int(b_str)
                except (ValueError, AttributeError):
                    continue
                if a in ends or b in ends:
                    matches.append(obj)

            if not matches:
                print("[Game] No active standing domino on World Map matches open ends.")
                self.parent.match_target = None
                self.parent.chain_open_target_obj = None
                self.parent.target_labels = set()
                self.post_failure()
                return

            target = min(
                matches,
                key=lambda o: math.hypot(o.pose.x - self.robot.pose.x,
                                        o.pose.y - self.robot.pose.y),
            )

            label_parts = target.face_label.split("-")
            if len(label_parts) == 2:
                self.parent.target_labels = {target.face_label, f"{label_parts[1]}-{label_parts[0]}"}
            else:
                self.parent.target_labels = {target.face_label}

            match_number = None
            if len(label_parts) == 2:
                try:
                    a, b = int(label_parts[0]), int(label_parts[1])
                    if a in ends:
                        match_number = a
                    elif b in ends:
                        match_number = b
                except ValueError:
                    match_number = None
            self.parent.match_number = match_number

            # Correct assignment for open/inner pips based on target ends
            try:
                b_p1, b_p2 = map(int, chain_end_obj.face_label.split("-"))
                if b_p1 in ends and b_p1 == match_number:
                    self.parent.b_open_pip = b_p1
                    self.parent.b_inner_pip = b_p2
                elif b_p2 in ends and b_p2 == match_number:
                    self.parent.b_open_pip = b_p2
                    self.parent.b_inner_pip = b_p1
                else:
                    self.parent.b_open_pip = match_number
                    self.parent.b_inner_pip = b_p1 if b_p2 == match_number else b_p2
            except Exception:
                self.parent.b_open_pip = match_number
                self.parent.b_inner_pip = match_number

            print(f"[Game] Matching target found on Live Map: {target.id} ({target.face_label})")
            print(f"[Game] Target Open End Chain Object: {chain_end_obj.id} at ({chain_end_obj.pose.x:.1f}, {chain_end_obj.pose.y:.1f})")
            self.parent.match_target = target
            self.post_success()

    class WaitHumanTurn(StateNode):
        def start(self, event=None):
            super().start(event)
            print("\n" + "=" * 50)
            print("[Game] Pass, play your move")
            print("=" * 50)

            while True:
                user_inp = input("Type 'done' when you have placed your domino: ").strip().lower()
                if user_inp == "done":
                    break
                print("Invalid input. Type 'done' to continue.")

            print("[Game] Human move recorded. Rescanning fallen dominoes...\n")
            self.post_completion()

    class TurnToTarget(ActionNode):
        def start(self, event=None):
            super().start(event)
            target = self.parent.match_target
            if target is None:
                self.post_failure()
                return

            self.parent.return_pose = (
                self.robot.pose.x,
                self.robot.pose.y,
                self.robot.pose.theta,
            )
            print(f"[Game] Recorded return-home pose: "
                  f"x={self.parent.return_pose[0]:.1f}mm, "
                  f"y={self.parent.return_pose[1]:.1f}mm, "
                  f"theta={math.degrees(self.parent.return_pose[2]):.1f}deg")

            dx = target.pose.x - self.robot.pose.x
            dy = target.pose.y - self.robot.pose.y

            r_theta = self.robot.pose.theta
            dx_b = dx * math.cos(r_theta) + dy * math.sin(r_theta)
            dy_b = -dx * math.sin(r_theta) + dy * math.cos(r_theta)

            self.parent._pending_distance_mm = dx_b
            self.parent._pending_sideways_mm = dy_b

            print(f"[Game] Prepared orthogonal approach: backing up {dx_b:.1f}mm then sideways shift {dy_b:.1f}mm")

    class DriveToTarget(ActionNode):
        def start(self, event=None):
            super().start(event)
            sideways_mm = self.parent._pending_sideways_mm
            print(f"[Game] Translating sideways {sideways_mm:.1f}mm to line up with target domino")
            self.robot.actuators["drive"].sideways(self, sideways_mm, None)

    class KprResetRun(StateNode):
        def start(self, event=None):
            super().start(event)
            self.parent.turn_steps = 0
            self.parent.target_data = None
            self.parent.last_charge_distance_mm = 0.0
            self.parent.actual_retreated_mm = 0.0
            self.parent.a_match_side = None
            self.parent.a_needs_flip = False
            self.parent.measured_forward_cm = 0.0
            self.parent.measured_lateral_cm = 0.0
            self.parent.measured_dividing_angle_deg = 0.0
            self.parent.post_turn_sideways_mm = 0.0
            self.parent.post_turn_angle_rad = 0.0
            self.post_completion()

    class TrackAndIdentify(StateNode):
        def start(self, event=None):
            super().start(event)
            frame = self.robot.camera_image

            if frame is None or self.parent.segmenter is None:
                self.parent.target_data = None
                self.post_failure()
                return

            image_h, image_w = frame.shape[:2]
            results = self.parent.segmenter(frame, conf=0.25, iou=0.5, agnostic_nms=True, verbose=False)

            if not results:
                self.parent.target_data = None
                self.post_failure()
                return

            result = results[0]
            candidates = []
            obb = getattr(result, "obb", None)
            boxes = getattr(result, "boxes", None)
            quads_info = []
            detected_indexes = set()

            if obb is not None and len(obb) > 0:
                raw_quads = list(obb.xyxyxyxy.cpu().numpy())
                for idx, q in enumerate(raw_quads):
                    quads_info.append((q, None))
                    detected_indexes.add(idx)

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                for idx, box in enumerate(xyxy):
                    if idx not in detected_indexes:
                        x1b, y1b, x2b, y2b = box
                        quad = np.array([[x1b, y1b], [x2b, y1b], [x2b, y2b], [x1b, y2b]], dtype=np.float32)
                        pixel_length = max(x2b - x1b, y2b - y1b)
                        quads_info.append((quad, pixel_length))

            for raw_obb, fallback_length in quads_info:
                crop, bbox, pixel_length = self.parent.rotate_crop(frame, raw_obb)
                if pixel_length <= 0 and fallback_length is not None:
                    pixel_length = fallback_length

                if crop is None or crop.size == 0 or pixel_length <= 0:
                    continue

                h, w, _ = crop.shape
                if h < 50 or w < 50:
                    continue

                left_raw, right_raw = self.parent.split_halves(crop)
                if left_raw is None or right_raw is None:
                    continue

                left_clean = self.parent.remove_white_border(left_raw)
                right_clean = self.parent.remove_white_border(right_raw)

                if left_clean is None or right_clean is None or left_clean.size == 0 or right_clean.size == 0:
                    continue

                left_tensor = self.parent.preprocess_half(left_clean)
                right_tensor = self.parent.preprocess_half(right_clean)

                if left_tensor is None or right_tensor is None:
                    continue

                left_pred, left_conf = self.parent.predict_half(left_tensor)
                right_pred, right_conf = self.parent.predict_half(right_tensor)

                if left_pred is None or right_pred is None:
                    continue

                x1_box, y1_box, x2_box, y2_box = bbox
                cx = float((x1_box + x2_box) * 0.5)

                divider_x = estimate_divider_x(frame, x1_box, y1_box, x2_box, y2_box)
                aim_x = float(divider_x) if divider_x is not None else cx

                angle_deg, distance_cm = estimate_angle_and_distance_cm(aim_x, image_w, pixel_length)
                aim_error_px = float(aim_x - (image_w * 0.5))

                candidates.append({
                    "bbox": bbox,
                    "label": f"{left_pred}-{right_pred}",
                    "confidence": min(left_conf, right_conf),
                    "cx": cx,
                    "aim_x": aim_x,
                    "cy": float((y1_box + y2_box) * 0.5),
                    "angle_deg": angle_deg,
                    "distance_cm": distance_cm,
                    "aim_error_px": aim_error_px,
                    "using_black_line": (divider_x is not None),
                    "left_pred": left_pred,
                    "right_pred": right_pred
                })

            if not candidates:
                self.parent.target_data = None
                self.post_failure()
                return

            view = frame.copy()
            valid_targets = []
            target_labels = self.parent.target_labels

            for cand in candidates:
                bx1, by1, bx2, by2 = map(int, cand["bbox"])
                if target_labels and cand["label"] in target_labels:
                    box_color = (0, 255, 0)
                    label_text = f"TARGET: {cand['label']} ({cand['confidence']:.2f})"
                    valid_targets.append(cand)
                else:
                    box_color = (0, 0, 255)
                    label_text = f"IGNORE: {cand['label']}"

                cv2.rectangle(view, (bx1, by1), (bx2, by2), box_color, 2)
                aim_px = int(cand["aim_x"])
                cv2.line(view, (aim_px, by1), (aim_px, by2), (255, 0, 255), 2)
                cv2.putText(view, f"{label_text} - {cand['distance_cm']:.1f}cm", (bx1, max(15, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

            imshow("Multi-Target Tracking Feed", view)

            if not target_labels or not valid_targets:
                self.parent.target_data = None
                self.post_failure()
                return

            best_target = max(valid_targets, key=lambda c: c["cy"])
            self.parent.target_data = best_target
            self.parent.a_near_pip = best_target.get("left_pred")
            self.parent.a_far_pip = best_target.get("right_pred")

            self.post_data(best_target)

    class ApproachCheckpoint(ActionNode):
        def start(self, event=None):
            super().start(event)
            target = self.parent.target_data
            if target is None:
                self.post_failure()
                return

            total_distance_mm = float(target["distance_cm"]) * 10.0
            self.parent.last_charge_distance_mm = total_distance_mm

            approach_distance_mm = max(0.0, total_distance_mm - 20.0)
            print(f"[Approach Checkpoint] Driving {approach_distance_mm:.1f}mm to reach 2cm from domino...")
            self.robot.actuators["drive"].forward(self, approach_distance_mm)

    class TurnRight45(ActionNode):
        def start(self, event=None):
            super().start(event)
            print("[Knock Sequence 1/4] Turning 45 degrees RIGHT...")
            self.robot.actuators["drive"].turn(self, math.radians(-45.0), None)

    class ShiftRight2cm(ActionNode):
        def start(self, event=None):
            super().start(event)
            print(f"[Knock Sequence 2/4] Moving {KNOCK_SIDEWAYS_MM:.1f} mm (2 cm) RIGHT sideways...")
            self.robot.actuators["drive"].sideways(self, KNOCK_SIDEWAYS_MM, None)

    class ShiftLeft2cm(ActionNode):
        def start(self, event=None):
            super().start(event)
            print(f"[Knock Sequence 3/4] Moving {-KNOCK_SIDEWAYS_MM:.1f} mm (-2 cm) LEFT sideways...")
            self.robot.actuators["drive"].sideways(self, -KNOCK_SIDEWAYS_MM, None)

    class TurnLeft45(ActionNode):
        def start(self, event=None):
            super().start(event)
            print("[Knock Sequence 4/4] Turning 45 degrees LEFT (reversing rotation)...")
            print("[Knock Sequence 4/4] Driving +7 cm forward, then backing up -3 cm...")
            self.robot.actuators["drive"].forward(self, 70.0, None)
            self.robot.actuators["drive"].forward(self, -30.0, None)
            self.parent.actual_retreated_mm = KNOCK_RETREAT_MM
            self.robot.actuators["drive"].turn(self, math.radians(45.0), None)

    # ==========================================
    # FALLEN DOMINO MEASUREMENT ROUTINE
    # ==========================================
    class PreparePickupRun(StateNode):
        def start(self, event=None):
            super().start(event)
            print("\n==============================================")
            print(">>> [SCANNING FALLEN DOMINO POSE & DISTANCE] <<<")
            print("==============================================")
            self.parent.target = None
            self.parent.debug_view = None
            self.post_completion()

    class DetectTarget(StateNode):
        def start(self, event=None):
            super().start(event)
            if self.parent.fallen_model is None:
                print("[FALLEN-DETECT] ERROR: Fallen model is missing/None!")
                self.post_failure()
                return

            time.sleep(SETTLE_DELAY_SEC)

            image = None
            results = []
            for attempt in range(DETECT_TRIES):
                image = self.robot.camera_image
                if image is None:
                    time.sleep(DETECT_RETRY_DELAY_SEC)
                    continue

                canvas, scale, pad_x, pad_y, new_w, new_h = roboflow_fit_resize(image, MODEL_SIZE)
                try:
                    results = self.parent.fallen_model.predict(
                        source=canvas,
                        conf=PREDICT_CONF,
                        iou=PREDICT_IOU,
                        verbose=False,
                    )
                except Exception as ex:
                    print(f"[FALLEN-DETECT] Predict failed: {ex}")
                    results = []

                if len(results) > 0 and results[0].masks is not None and len(results[0].masks.data) > 0:
                    break
                time.sleep(DETECT_RETRY_DELAY_SEC)

            if image is None or len(results) == 0 or results[0].masks is None:
                print("[FALLEN-DETECT] FAILED: No fallen domino detected.")
                self.post_failure()
                return

            h, w = image.shape[:2]
            view = image.copy()

            result = results[0]
            mask_data = result.masks.data
            candidates = []

            for i in range(len(mask_data)):
                mask_small = mask_data[i].detach().cpu().numpy()
                if np.count_nonzero(mask_small > 0.1) == 0:
                    continue

                mask_orig = map_mask_to_original(mask_small, pad_x, pad_y, new_w, new_h, w, h)
                mask_area = float(cv2.countNonZero(mask_orig))
                contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    small_contours, _ = cv2.findContours((mask_small > 0.1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not small_contours:
                        continue
                    cnt_small = max(small_contours, key=cv2.contourArea)
                    rect_small = cv2.minAreaRect(cnt_small)
                    (cx_s, cy_s), (rw_s, rh_s), ang_s = rect_small
                    cx = (cx_s - pad_x) / scale
                    cy = (cy_s - pad_y) / scale
                    rect = ((cx, cy), (rw_s / scale, rh_s / scale), ang_s)
                    mask_area = float(cv2.contourArea(cnt_small)) / (scale * scale)
                else:
                    cnt = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(cnt)
                    (cx, cy), _, _ = rect

                candidates.append({
                    "mask_area": mask_area,
                    "cx": cx,
                    "cy": cy,
                    "rect": rect
                })

            if len(candidates) == 0:
                print("[FALLEN-DETECT] FAILED: Zero mask candidates.")
                self.post_failure()
                return

            best = max(candidates, key=lambda c: c["mask_area"])
            _, _, orientation_deg, long_side_px = long_axis_from_rect(best["rect"])

            try:
                center_x_mm, center_y_mm = project_pixel_to_base(
                    self.robot, 
                    best["cx"], 
                    best["cy"], 
                    bbox_long_side_px=long_side_px, 
                    frame_width=w
                )
            except Exception as ex:
                print(f"[FALLEN-DETECT] Projection Error: {ex}")
                self.post_failure()
                return

            retreat_cm = self.parent.actual_retreated_mm / 10.0
            camera_forward_cm = center_x_mm / 10.0
            
            self.parent.measured_lateral_cm = center_y_mm / 10.0
            self.parent.measured_forward_cm = retreat_cm + camera_forward_cm
            self.parent.measured_dividing_angle_deg = wrap_deg(orientation_deg)

            total_distance_cm = math.hypot(self.parent.measured_forward_cm, self.parent.measured_lateral_cm)

            print("\n" + "="*50)
            print("        FALLEN DOMINO DETECTION METRICS        ")
            print("="*50)
            print(f" ► Direct Distance : {total_distance_cm:.2f} cm")
            print(f" ► Forward Distance: {self.parent.measured_forward_cm:.2f} cm")
            print(f" ► Lateral Offset  : {self.parent.measured_lateral_cm:.2f} cm")
            print(f" ► Dividing Angle  : {self.parent.measured_dividing_angle_deg:.2f}°")
            print("="*50 + "\n")

            box = cv2.boxPoints(best["rect"]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(view, [box], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.circle(view, (int(best["cx"]), int(best["cy"])), 4, (255, 0, 0), -1)
            imshow("fallen_domino_measurement", view)

            self.post_success()

    # ==========================================
    # INTERMEDIATE PUSH & RETREAT ROUTINE
    # ==========================================
    class MidStep_MoveLateralHalf(ActionNode):
        """Move by Measured Lateral Offset / 2 with adjustment"""
        def start(self, event=None):
            super().start(event)
            lat_half_mm = -(self.parent.measured_lateral_cm * 10.0) / 2.0 + 10.0# Add 2mm buffer for safety
            print(f"\n[INTERMEDIATE 1/3] Moving Lateral Offset / 2: {lat_half_mm:.1f}mm")
            self.robot.actuators["drive"].sideways(self, lat_half_mm, None)

    class MidStep_MoveForwardPlus12cm(ActionNode):
        """Move Measured Forward Distance + 12 cm (120mm)"""
        def start(self, event=None):
            super().start(event)
            fwd_target_mm = (self.parent.measured_forward_cm * 10.0) + 70.0
            print(f"[INTERMEDIATE 2/3] Moving Forward (Measured + 12cm): {fwd_target_mm:.1f}mm")
            self.robot.actuators["drive"].forward(self, fwd_target_mm, None)

    class MidStep_Retreat3cm(ActionNode):
        """Retreat 3 cm (30mm) back again before re-measurement"""
        def start(self, event=None):
            super().start(event)
            print("[INTERMEDIATE 3/3] Retreating 3 cm (30mm)...")
            self.parent.actual_retreated_mm = 30.0
            self.robot.actuators["drive"].forward(self, -30.0, None)

    class ClassifyManipulationCase(StateNode):
        def start(self, event=None):
            super().start(event)

            match_pip = self.parent.match_number

            if (self.parent.a_near_pip == 0 and self.parent.a_far_pip == 0) or (self.parent.a_near_pip is None):
                if self.parent.match_target and self.parent.match_target.face_label:
                    try:
                        parts = self.parent.match_target.face_label.split("-")
                        p1, p2 = int(parts[0]), int(parts[1])
                        self.parent.a_near_pip = p1
                        self.parent.a_far_pip = p2
                    except Exception:
                        pass

            a_near = self.parent.a_near_pip
            a_far = self.parent.a_far_pip
            b_open = self.parent.b_open_pip
            b_inner = self.parent.b_inner_pip

            print(f"[CASE DEBUG] A_near:{a_near}, A_far:{a_far} | B_open:{b_open}, B_inner:{b_inner} | Match:{match_pip}")

            # STRICT DRILL MATRIX ROUTING:
            if a_near == match_pip and b_inner == match_pip:
                self.parent.manipulation_case = 1
                print("[CASE CLASSIFICATION] Selected CASE 1: Knocked (4,3) -> Fallen (4,1)")
                self.post_data("case_1")

            elif a_near == match_pip and b_open == match_pip:
                self.parent.manipulation_case = 2
                print("[CASE CLASSIFICATION] Selected CASE 2: Knocked (4,3) -> Fallen (1,4)")
                self.post_data("case_2")

            elif a_far == match_pip and b_open == match_pip:
                self.parent.manipulation_case = 3
                print("[CASE CLASSIFICATION] Selected CASE 3: Knocked (3,4) -> Fallen (1,4)")
                self.post_data("case_3")

            elif a_far == match_pip and b_inner == match_pip:
                self.parent.manipulation_case = 4
                print("[CASE CLASSIFICATION] Selected CASE 4: Knocked (3,4) -> Fallen (4,1)")
                self.post_data("case_4")

            else:
                # Primary fallback routing based on matching pip position
                if a_far == match_pip or (a_near != match_pip and b_open == match_pip):
                    self.parent.manipulation_case = 3
                    print("[CASE CLASSIFICATION] Fallback -> Selected CASE 3")
                    self.post_data("case_3")
                else:
                    self.parent.manipulation_case = 2
                    print("[CASE CLASSIFICATION] Fallback -> Selected CASE 2")
                    self.post_data("case_2")

            self.post_completion()

    # ------------------------------------------------------------------
    # Case Routine Building Blocks
    # ------------------------------------------------------------------

    class _SidewaysToKnockedDomino(ActionNode):
        """Sideways move shifting towards the NON-MATCHING side of the knocked domino."""
        def start(self, event=None):
            super().start(event)
            case_num = self.parent.manipulation_case
            
            # Case 3 and Case 4 are ALWAYS positive (+ve)
            if case_num in (3, 4):
                dist = abs(self.parent.la_mm) + 100.0
            # Case 1 and Case 2 are ALWAYS negative (-ve)
            else:
                dist = -(abs(self.parent.la_mm) + 100.0)

            print(f"[Case Move] Sideways {dist:.1f}mm toward non-matching end of knocked domino")
            self.robot.actuators["drive"].sideways(self, dist, None)

    class _ForwardToKnockedDomino(ActionNode):
        """Forward move by true camera distance F_A + 2 cm buffer."""
        def start(self, event=None):
            super().start(event)
            dist = max(0.0, self.parent.fa_mm + 100.0)
            print(f"[Case Move] Forward {dist:.1f}mm to knocked domino")
            self.robot.actuators["drive"].forward(self, dist, None)

    class _TurnComplementaryAngle(ActionNode):
        """Turn by complementary angle (90 deg - orientation) with sign based on case."""
        def start(self, event=None):
            super().start(event)
            case_num = self.parent.manipulation_case
            raw_comp = abs(90.0 - self.parent.theta_a_deg)
            
            # Cases 1 and 2 are ALWAYS positive (+ve)
            if case_num in (1, 2):
                comp_deg = wrap_deg(raw_comp)
            # Cases 3 and 4 are ALWAYS negative (-ve)
            else:
                comp_deg = wrap_deg(-raw_comp)

            print(f"[Case Move] Turning complementary angle {comp_deg:.1f} deg (Case {case_num})")
            self.robot.actuators["drive"].turn(self, math.radians(comp_deg), None)

    class _ForwardToChainLateralPlusBuffer(ActionNode):
        """Forward move incorporating sideways offset compensation plus buffer."""
        def start(self, event=None):
            super().start(event)
            sideways_offset = abs(self.parent.la_mm) + 100.0
            dist = self.parent.lb_mm + 100.0 + sideways_offset
            print(f"[Case Move] Forward {dist:.1f}mm toward chain target (lateral + buffer + compensated offset)")
            self.robot.actuators["drive"].forward(self, dist, None)

    class _ForwardToChainLateral(ActionNode):
        """Forward move incorporating sideways offset compensation."""
        def start(self, event=None):
            super().start(event)
            sideways_offset = abs(self.parent.la_mm) + 100.0
            dist = self.parent.lb_mm + 50.0 + sideways_offset
            print(f"[Case Move] Forward {dist:.1f}mm toward chain target (lateral + compensated offset)")
            self.robot.actuators["drive"].forward(self, dist, None)

    class _SemicircleArcToChain(ActionNode):
        """Semicircle arc with radius sign based on manipulation case:
        - Case 1 and Case 4: Negative (-ve) radius
        - Case 2 and Case 3: Positive (+ve) radius
        """
        def start(self, event=None):
            super().start(event)
            case_num = self.parent.manipulation_case
            mag_radius = max(5.0, (self.parent.fb_mm / 2.0)+33.0)

            if case_num in (1, 4):
                signed_radius = -abs(mag_radius)
            else:  # Case 2 and Case 3
                signed_radius = abs(mag_radius)

            print(f"[Case Move] Semicircle arc (Case {case_num}): radius {signed_radius:.1f}mm")
            self.robot.actuators["drive"].drive_arc(self, signed_radius, math.pi, None, 1.0)

    class _SemicircleArcToChain2(ActionNode):
            """Semicircle arc with radius sign based on manipulation case:
            - Case 1 and Case 4: Negative (-ve) radius
            - Case 2 and Case 3: Positive (+ve) radius
            """
            def start(self, event=None):
                super().start(event)
                case_num = self.parent.manipulation_case
                mag_radius = max(5.0, (self.parent.fb_mm / 2.0)-33.0)
    
                if case_num in (1, 4):
                    signed_radius = -abs(mag_radius)
                else:  # Case 2 and Case 3
                    signed_radius = abs(mag_radius)
    
                print(f"[Case Move] Semicircle arc (Case {case_num}): radius {signed_radius:.1f}mm")
                self.robot.actuators["drive"].drive_arc(self, signed_radius, math.pi, None, 1.0)

    class _RealignToChainAngle(ActionNode):
        """Re-aligns the robot parallel to the target fallen domino's axis 
        using the World Map object pose with minimal-turn 180-degree symmetry.
        """
        def start(self, event=None):
            super().start(event)
            print("[Re-Align] Semicircle arc complete. Referencing World Map for target angle...")
            time.sleep(SETTLE_DELAY_SEC)

            chain_obj = self.parent.chain_open_target_obj
            if chain_obj is None or getattr(chain_obj, "pose", None) is None:
                print("[Re-Align] Target domino missing from World Map; keeping current heading.")
                self.post_completion()
                return

            target_theta_rad = chain_obj.pose.theta
            robot_theta_rad = self.robot.pose.theta

            # Calculate raw angular difference
            raw_delta = normalize_axis_angle(target_theta_rad - robot_theta_rad)
            raw_deg = math.degrees(raw_delta)

            # Enforce 180-degree line symmetry for minimal rotation
            delta_deg = wrap_deg(raw_deg)
            if delta_deg > 90.0:
                delta_deg -= 180.0
            elif delta_deg < -90.0:
                delta_deg += 180.0

            print(f"[Re-Align] Target domino WorldMap theta: {math.degrees(target_theta_rad):.1f}° | Robot theta: {math.degrees(robot_theta_rad):.1f}°")
            print(f"[Re-Align] Minimal symmetric alignment turn: {delta_deg:.1f}°")

            if abs(delta_deg) > MIN_TURN_DEG:
                print(f"[Re-Align] Executing correction turn: {delta_deg:.1f}°")
                self.robot.actuators["drive"].turn(self, math.radians(delta_deg), None)
                return

            print("[Re-Align] Robot is parallel to target domino black line.")
            self.post_completion()

    class _ForwardLevelWithChain(ActionNode):
        def start(self, event=None):
            super().start(event)
            dist = self.parent.fb_mm - self.parent.fa_mm
            print(f"[Case Move] Forward {dist:.1f}mm to level with chain target")
            self.robot.actuators["drive"].forward(self, dist, None)

    class _FlipArc(ActionNode):
        def start(self, event=None):
            super().start(event)
            print("[Case Move] Flip arc: radius 10.0mm, pi rad")
            self.robot.actuators["drive"].drive_arc(self, 10.0, math.pi, None, 1.0)

    class _DockToChainExtended(ActionNode):
        """Extended forward dock for Flip Arc routines (Case 2 & Case 4).
        Covers the extra gap: 10 cm (100 mm) + L_B (lateral chain offset).
        """
        def start(self, event=None):
            super().start(event)
            dock_dist = 100.0 + max(0.0, abs(self.parent.lb_mm))
            print(f"[Case Move] Extended forward dock: {dock_dist:.1f}mm toward target (10cm + L_B)")
            self.robot.actuators["drive"].forward(self, dock_dist, None)

    def _build_case_chain(self, prefix, node_specs, wait_scan_node):
        nodes = []
        for suffix, node in node_specs:
            node.set_name(f"{prefix}_{suffix}").set_parent(self)
            nodes.append(node)
        for a, b in zip(nodes, nodes[1:]):
            CompletionTrans().add_sources(a).add_destinations(b)
            FailureTrans().add_sources(a).add_destinations(wait_scan_node)
        FailureTrans().add_sources(nodes[-1]).add_destinations(wait_scan_node)
        return nodes[0], nodes[-1]

    def _build_case1_chain(self, wait_scan_node):
        """Case 1: Knocked (4,3) -> Fallen (4,1)"""
        return self._build_case_chain("case1", [
            ("sideways_approach", self._SidewaysToKnockedDomino()),
            ("forward_approach", self._ForwardToKnockedDomino()),
            ("turn_complementary", self._TurnComplementaryAngle()),
            ("grab", self.ForwardMove(30.0)),
            ("forward_to_chain", self._ForwardToChainLateralPlusBuffer()),
            ("semicircle", self._SemicircleArcToChain()),
            ("realign", self._RealignToChainAngle()),
            ("dock", self.ForwardMove(30.0)),
        ], wait_scan_node)

    def _build_case2_chain(self, wait_scan_node):
        """Case 2: Knocked (4,3) -> Fallen (1,4) using Flip Arc strategy."""
        return self._build_case_chain("case2", [
            ("sideways_approach", self._SidewaysToKnockedDomino()),
            ("forward_approach", self._ForwardToKnockedDomino()),
            ("turn_complementary", self._TurnComplementaryAngle()),
            ("grab", self.ForwardMove(30.0)),
            ("flip_arc", self._FlipArc()),
            ("forward_to_chain", self._ForwardToChainLateralPlusBuffer()),
            ("semicircle", self._SemicircleArcToChain2()),
            ("realign", self._RealignToChainAngle()),
            ("dock", self._DockToChainExtended()),
        ], wait_scan_node)

    def _build_case3_chain(self, wait_scan_node):
        """Case 3: Knocked (3,4) -> Fallen (1,4)"""
        return self._build_case_chain("case3", [
            ("sideways_approach", self._SidewaysToKnockedDomino()),
            ("forward_approach", self._ForwardToKnockedDomino()),
            ("turn_complementary", self._TurnComplementaryAngle()),
            ("forward_to_chain", self._ForwardToChainLateral()),
            ("semicircle", self._SemicircleArcToChain()),
            ("realign", self._RealignToChainAngle()),
            ("dock", self.ForwardMove(30.0)),
        ], wait_scan_node)

    def _build_case4_chain(self, wait_scan_node):
        """Case 4: Knocked (3,4) -> Fallen (4,1)"""
        return self._build_case_chain("case4", [
            ("sideways_approach", self._SidewaysToKnockedDomino()),
            ("forward_approach", self._ForwardToKnockedDomino()),
            ("turn_complementary", self._TurnComplementaryAngle()),
            ("grab", self.ForwardMove(30.0)),
            ("flip_arc", self._FlipArc()),
            ("forward_to_chain", self._ForwardToChainLateralPlusBuffer()),
            ("semicircle", self._SemicircleArcToChain()),
            ("realign", self._RealignToChainAngle()),
            ("dock", self._DockToChainExtended()),
        ], wait_scan_node)

    class Step5_FinalKick(ActionNode):
        def start(self, event=None):
            super().start(event)
            print("[Placement] Matching sides connected! Executing PlaceKick().now()...")
            try:
                PlaceKick().now()
            except Exception as e:
                print(f"[Placement] PlaceKick execution warning: {e}")
            
            print("[Placement] Backing up straight -6 cm (-60 mm) in current heading post-kick...")
            self.robot.actuators["drive"].forward(self, -70.0, None)

            drive_actuator = self.robot.actuators.get("drive")
            if drive_actuator and drive_actuator.holder is not None:
                drive_actuator.unlock(drive_actuator.holder)

    # ------------------------------------------------------------------
    # RETURN-HOME & POST-MOVE nodes
    # ------------------------------------------------------------------

    class TurnToReturnPose(ActionNode):
        def start(self, event=None):
            super().start(event)
            if self.parent.return_pose is None:
                print("[Return] No return_pose recorded; skipping return-home turn.")
                self.post_failure()
                return

            rx, ry, _rtheta = self.parent.return_pose
            dx = rx - self.robot.pose.x
            dy = ry - self.robot.pose.y
            self.parent._return_distance_mm = math.hypot(dx, dy)
            bearing = normalize_axis_angle(math.atan2(dy, dx) - self.robot.pose.theta)
            print(f"[Return] Turning {math.degrees(bearing):.1f} deg toward start pose "
                  f"({self.parent._return_distance_mm:.1f}mm away)")
            self.robot.actuators["drive"].turn(self, bearing, None)

    class DriveToReturnPose(ActionNode):
        def start(self, event=None):
            super().start(event)
            dist = self.parent._return_distance_mm
            print(f"[Return] Driving {dist:.1f}mm back toward start pose")
            self.robot.actuators["drive"].forward(self, dist, None)

    class RestoreOriginalHeading(ActionNode):
        def start(self, event=None):
            super().start(event)
            if self.parent.return_pose is None:
                self.post_completion()
                return
            _rx, _ry, target_theta = self.parent.return_pose
            delta = normalize_axis_angle(target_theta - self.robot.pose.theta)
            print(f"[Return] Restoring original heading: turning {math.degrees(delta):.1f} deg")
            self.robot.actuators["drive"].turn(self, delta, None)

    class DriveForward15cm(ActionNode):
        def start(self, event=None):
            super().start(event)
            print(f"[Post-Return] Driving {POST_RETURN_FORWARD_MM:.1f} mm (15 cm) FORWARD...")
            self.robot.actuators["drive"].forward(self, POST_RETURN_FORWARD_MM, None)

    class ShiftRight10cm(ActionNode):
        def start(self, event=None):
            super().start(event)
            print(f"[Post-Return] Moving {POST_RETURN_RIGHT_MM:.1f} mm (10 cm) RIGHT sideways...")
            self.robot.actuators["drive"].sideways(self, POST_RETURN_RIGHT_MM, None)

    class ClearFallenAndMatchedFromWorldMap(StateNode):
        def start(self, event=None):
            super().start(event)
            world_map = self.robot.world_map
            target = self.parent.match_target
            target_label = target.face_label.split(".")[0] if target and target.face_label else None

            removed_keys = []
            with world_map._lock:
                keys_to_delete = []
                for k, obj in world_map.objects.items():
                    if isinstance(obj, DominoObj):
                        if getattr(obj, "is_fallen", False):
                            keys_to_delete.append(k)
                        elif target_label and target_label in k:
                            keys_to_delete.append(k)

                for k in keys_to_delete:
                    del world_map.objects[k]
                    removed_keys.append(k)

                if hasattr(world_map, "dispatch_update"):
                    world_map.dispatch_update()

            print(f"[WorldMap Clean] Erased {len(removed_keys)} object(s) from World Map: {removed_keys}")
            self.post_completion()

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _build_sweep_chain(self, prefix: str, expected_fallen: bool):
        steps = self._sweep_plan_mm()
        nodes = []
        prev_scan = None
        for i, dist in enumerate(steps):
            mover = self.SidewaysMove(distance_mm=dist) \
                .set_name(f"{prefix}_move{i}").set_parent(self)
            scanner = self.ScanAndPlace(expected_fallen=expected_fallen) \
                .set_name(f"{prefix}_scan{i}").set_parent(self)

            CompletionTrans().add_sources(mover).add_destinations(scanner)
            FailureTrans().add_sources(mover).add_destinations(scanner)

            if prev_scan is not None:
                CompletionTrans().add_sources(prev_scan).add_destinations(mover)

            nodes.append(mover)
            prev_scan = scanner
        return nodes[0], prev_scan

    def setup(self):
        intro = Print("[Game] Starting domino chain game...") \
            .set_name("intro").set_parent(self)
        reset = self.ResetGame().set_name("reset").set_parent(self)

        std_first, std_last = self._build_sweep_chain("std", expected_fallen=False)
        report_std = self.ReportStanding().set_name("report_std").set_parent(self)

        reposition1 = self.SidewaysMove(self._sideways_left(REPOSITION_LEFT_MM)) \
            .set_name("reposition_left").set_parent(self)
        reposition2 = self.ForwardMove(REPOSITION_FORWARD_MM) \
            .set_name("reposition_fwd").set_parent(self)
        reposition3 = self.SidewaysMove(self._sideways_right(REPOSITION_LEFT_MM)) \
            .set_name("reposition_right").set_parent(self)

        fallen_first, fallen_last = self._build_sweep_chain("fallen", expected_fallen=True)
        analyze = self.AnalyzeChain().set_name("analyze").set_parent(self)

        human_wait = self.WaitHumanTurn().set_name("human_wait").set_parent(self)

        return1 = self.SidewaysMove(self._sideways_left(REPOSITION_LEFT_MM)) \
            .set_name("return_left").set_parent(self)
        return2 = self.ForwardMove(-(REPOSITION_FORWARD_MM + 50.0)) \
            .set_name("return_back").set_parent(self)
        return3 = self.SidewaysMove(self._sideways_right(REPOSITION_LEFT_MM)) \
            .set_name("reposition_right").set_parent(self)

        turn_target = self.TurnToTarget().set_name("turn_target").set_parent(self)
        drive_target = self.DriveToTarget().set_name("drive_target").set_parent(self)

        kpr_reset = self.KprResetRun().set_name("kpr_reset").set_parent(self)
        kpr_track = self.TrackAndIdentify().set_name("kpr_track").set_parent(self)

        kpr_approach_chk = self.ApproachCheckpoint().set_name("kpr_approach_chk").set_parent(self)

        # Knock Sequence
        kpr_turn_right45 = self.TurnRight45().set_name("kpr_turn_right45").set_parent(self)
        kpr_shift_right2cm = self.ShiftRight2cm().set_name("kpr_shift_right2cm").set_parent(self)
        kpr_shift_left2cm = self.ShiftLeft2cm().set_name("kpr_shift_left2cm").set_parent(self)
        kpr_turn_left45 = self.TurnLeft45().set_name("kpr_turn_left45").set_parent(self)

        prep_pickup = self.PreparePickupRun().set_name("prep_pickup").set_parent(self)
        kpr_detect = self.DetectTarget().set_name("kpr_detect").set_parent(self)

        mid_lat = self.MidStep_MoveLateralHalf().set_name("mid_lat").set_parent(self)
        mid_fwd = self.MidStep_MoveForwardPlus12cm().set_name("mid_fwd").set_parent(self)
        mid_ret = self.MidStep_Retreat3cm().set_name("mid_ret").set_parent(self)
        kpr_detect_retry = self.DetectTarget().set_name("kpr_detect_retry").set_parent(self)

        kpr_wait_scan = StateNode().set_name("kpr_wait_scan").set_parent(self)

        classify_case = self.ClassifyManipulationCase().set_name("classify_case").set_parent(self)
        case1_first, case1_last = self._build_case1_chain(kpr_wait_scan)
        case2_first, case2_last = self._build_case2_chain(kpr_wait_scan)
        case3_first, case3_last = self._build_case3_chain(kpr_wait_scan)
        case4_first, case4_last = self._build_case4_chain(kpr_wait_scan)

        step_final_kick = self.Step5_FinalKick().set_name("step_final_kick").set_parent(self)

        # Post-Move Return & Reset
        kpr_return_turn = self.TurnToReturnPose().set_name("kpr_return_turn").set_parent(self)
        kpr_return_drive = self.DriveToReturnPose().set_name("kpr_return_drive").set_parent(self)
        kpr_return_heading = self.RestoreOriginalHeading().set_name("kpr_return_heading").set_parent(self)

        kpr_forward_15cm = self.DriveForward15cm().set_name("kpr_forward_15cm").set_parent(self)
        kpr_shift_right_10cm = self.ShiftRight10cm().set_name("kpr_shift_right_10cm").set_parent(self)
        kpr_clear_map = self.ClearFallenAndMatchedFromWorldMap().set_name("kpr_clear_map").set_parent(self)

        kpr_done_print = Print("[Game] kickpickrefine sequence completed; passing turn to human.") \
            .set_name("kpr_done_print").set_parent(self)
        kpr_give_up = Print("[Game] Could not visually re-acquire matched domino; giving up this turn.") \
            .set_name("kpr_give_up").set_parent(self)

        # Transitions
        CompletionTrans().add_sources(intro).add_destinations(reset)
        CompletionTrans().add_sources(reset).add_destinations(std_first)

        CompletionTrans().add_sources(std_last).add_destinations(report_std)
        CompletionTrans().add_sources(report_std).add_destinations(reposition1)

        CompletionTrans().add_sources(reposition1).add_destinations(reposition2)
        FailureTrans().add_sources(reposition1).add_destinations(reposition2)
        CompletionTrans().add_sources(reposition2).add_destinations(reposition3)
        FailureTrans().add_sources(reposition2).add_destinations(reposition3)
        CompletionTrans().add_sources(reposition3).add_destinations(fallen_first)
        FailureTrans().add_sources(reposition3).add_destinations(fallen_first)

        CompletionTrans().add_sources(fallen_last).add_destinations(analyze)

        SuccessTrans().add_sources(analyze).add_destinations(return1)
        FailureTrans().add_sources(analyze).add_destinations(human_wait)

        # Human turn completed
        CompletionTrans().add_sources(human_wait).add_destinations(fallen_first)

        CompletionTrans().add_sources(return1).add_destinations(return2)
        FailureTrans().add_sources(return1).add_destinations(return2)
        CompletionTrans().add_sources(return2).add_destinations(return3)
        FailureTrans().add_sources(return2).add_destinations(return3)
        CompletionTrans().add_sources(return3).add_destinations(kpr_reset)
        FailureTrans().add_sources(return3).add_destinations(kpr_reset)
        
        CompletionTrans().add_sources(kpr_reset).add_destinations(kpr_track)
        DataTrans().add_sources(kpr_track).add_destinations(kpr_approach_chk)
        FailureTrans().add_sources(kpr_track).add_destinations(kpr_wait_scan)
        TimerTrans(1.0).add_sources(kpr_wait_scan).add_destinations(kpr_track)

        # Knock Sequence -> Prep -> Detect -> Intermediate Adjustments -> Remeasure -> Classify
        CompletionTrans().add_sources(kpr_approach_chk).add_destinations(kpr_turn_right45)
        FailureTrans().add_sources(kpr_approach_chk).add_destinations(kpr_wait_scan)

        CompletionTrans().add_sources(kpr_turn_right45).add_destinations(kpr_shift_right2cm)
        FailureTrans().add_sources(kpr_turn_right45).add_destinations(kpr_wait_scan)

        CompletionTrans().add_sources(kpr_shift_right2cm).add_destinations(kpr_shift_left2cm)
        FailureTrans().add_sources(kpr_shift_right2cm).add_destinations(kpr_wait_scan)

        CompletionTrans().add_sources(kpr_shift_left2cm).add_destinations(kpr_turn_left45)
        FailureTrans().add_sources(kpr_shift_left2cm).add_destinations(kpr_wait_scan)

        CompletionTrans().add_sources(kpr_turn_left45).add_destinations(prep_pickup)
        CompletionTrans().add_sources(prep_pickup).add_destinations(kpr_detect)

        # Intermediate adjustment routing loop (using SuccessTrans for DetectTarget)
        SuccessTrans().add_sources(kpr_detect).add_destinations(mid_lat)
        FailureTrans().add_sources(kpr_detect).add_destinations(kpr_done_print)

        CompletionTrans().add_sources(mid_lat).add_destinations(mid_fwd)
        FailureTrans().add_sources(mid_lat).add_destinations(kpr_done_print)

        CompletionTrans().add_sources(mid_fwd).add_destinations(mid_ret)
        FailureTrans().add_sources(mid_fwd).add_destinations(kpr_done_print)

        CompletionTrans().add_sources(mid_ret).add_destinations(kpr_detect_retry)
        FailureTrans().add_sources(mid_ret).add_destinations(kpr_done_print)

        # Case Router Transitions from remeasured target (using SuccessTrans for DetectTarget retry)
        SuccessTrans().add_sources(kpr_detect_retry).add_destinations(classify_case)
        FailureTrans().add_sources(kpr_detect_retry).add_destinations(kpr_done_print)

        DataTrans("case_1").add_sources(classify_case).add_destinations(case1_first)
        DataTrans("case_2").add_sources(classify_case).add_destinations(case2_first)
        DataTrans("case_3").add_sources(classify_case).add_destinations(case3_first)
        DataTrans("case_4").add_sources(classify_case).add_destinations(case4_first)

        # Case Completions -> Final Kick
        CompletionTrans().add_sources(case1_last).add_destinations(step_final_kick)
        CompletionTrans().add_sources(case2_last).add_destinations(step_final_kick)
        CompletionTrans().add_sources(case3_last).add_destinations(step_final_kick)
        CompletionTrans().add_sources(case4_last).add_destinations(step_final_kick)

        # Post-placement return sequence
        CompletionTrans().add_sources(step_final_kick).add_destinations(kpr_return_turn)
        FailureTrans().add_sources(kpr_return_turn).add_destinations(kpr_done_print)

        CompletionTrans().add_sources(kpr_return_turn).add_destinations(kpr_return_drive)
        FailureTrans().add_sources(kpr_return_drive).add_destinations(kpr_done_print)

        CompletionTrans().add_sources(kpr_return_drive).add_destinations(kpr_return_heading)
        FailureTrans().add_sources(kpr_return_heading).add_destinations(kpr_done_print)

        CompletionTrans().add_sources(kpr_return_heading).add_destinations(kpr_forward_15cm)
        CompletionTrans().add_sources(kpr_forward_15cm).add_destinations(kpr_shift_right_10cm)
        CompletionTrans().add_sources(kpr_shift_right_10cm).add_destinations(kpr_clear_map)
        CompletionTrans().add_sources(kpr_clear_map).add_destinations(kpr_done_print)

        NullTrans().add_sources(kpr_done_print).add_destinations(human_wait)
        NullTrans().add_sources(kpr_give_up).add_destinations(human_wait)

        return self