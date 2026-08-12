from __future__ import annotations

import os

os.environ["QT_LOGGING_RULES"] = (
    "qt.core.qobject.connect=false;"
    "qt.qpa.*=false;"
    "qt.scenegraph.*=false;"
    "*.debug=false"
)

import importlib
import math
import cv2
import numpy as np

from aim_fsm import *
from aim_fsm.worldmap import DominoObj, Pose
from aim_fsm.domino import DominoWorldDetector, normalize_axis_angle
from aim_fsm.pilot import PilotToPose

import aim_fsm.domino
import importlib
importlib.reload(aim_fsm.domino)

# ==========================================
# Calibration / tuning constants (Optimized for Speed)
# ==========================================
KNOWN_LENGTH_MM = 48.0
FOCAL_LENGTH = 396.3

APPROACH_STANDOFF_MM = 50.0  
KNOCK_TURN_DEG = 45.0         
KNOCK_NUDGE_MM = 20.0        
KNOCK_RETREAT_MM = 75.0      
MAX_KNOCK_ATTEMPTS = 3        
RESCAN_DELAY_SEC = 0.05      


class knock(StateMachineProgram):
    """runfsm("knock"): find a standing domino, register it on the world map,
    pilot up to it, and knock it over with a turn/nudge/nudge/turn-back swipe.
    Re-tries the swipe until the domino is detected as fallen, then reports
    success and moves on to look for the next standing domino."""

    def __init__(self):
        super().__init__(
            launch_cam_viewer=True,
            launch_worldmap_viewer=True,
            launch_path_viewer=True,
            speech=False,
            aruco=False,
            domino=False,
            domino_labeling=False,
            force_annotation=True,
        )

        detector = DominoWorldDetector(
            conf_threshold=0.35,  
            standing_weights="bestieee.pt",
            fallen_weights="fallen.pt",
            standing_label_weights="different.pt",
            fallen_label_weights="fallenhalf.pt",
            frame_skip=1,         
        )
        for attr in ["focal_length", "focal_length_px", "fx", "fy"]:
            if hasattr(detector, attr):
                setattr(detector, attr, FOCAL_LENGTH)

        self.robot.domino_detector = detector
        self._last_image = None

        self.target = None          
        self.target_id = None       
        self.knock_attempts = 0

    # ------------------------------------------------------------------
    # Camera callbacks (Only cache the image, no automatic world map updates)
    # ------------------------------------------------------------------
    def user_image(self, image, gray):
        detector = getattr(self.robot, "domino_detector", None)
        if detector is None or image is None:
            return
        self._last_image = image.copy()
        detector.detect(image, frame_id=getattr(self.robot, "frame_count", None))

    def user_annotate(self, image):
        out = image.copy()
        detector = getattr(self.robot, "domino_detector", None)
        if detector is None:
            return out

        for obs in detector.latest_observations():
            is_target = self.target_id is not None and obs.face_label == self.target_id.replace("domino_", "")
            color = (0, 165, 255) if is_target else ((255, 200, 0) if obs.is_fallen else (0, 230, 110))
            cx, cy = obs.center_xy
            status_str = "FALLEN" if obs.is_fallen else "STANDING"
            label = f"{obs.face_label or '?'} ({status_str})"
            if is_target:
                label = "TARGET " + label
            cv2.circle(out, (int(cx), int(cy)), 4, color, -1)
            cv2.putText(out, label, (int(cx) - 50, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # World-map bookkeeping
    # ------------------------------------------------------------------
    def _register_observations(self, observations):
        world_map = getattr(self.robot, "world_map", None)
        if world_map is None:
            return []

        updated_ids = []
        with world_map._lock if hasattr(world_map, "_lock") else nullcontext():
            for obs in observations:
                quad = np.array(obs.quad, dtype=np.float32) if getattr(obs, "quad", None) is not None else None
                cx, cy = obs.center_xy
                bottom_cy = float(np.max(quad[:, 1])) if quad is not None else cy
                hit, objpos = world_map.project_image_point_to_world(cx, bottom_cy)
                if objpos is None:
                    continue
                x_mm = float(objpos[0][0])
                y_mm = float(objpos[1][0])

                world_yaw = self.robot.pose.theta
                if quad is not None:
                    sorted_by_y = sorted(quad, key=lambda pt: pt[1], reverse=True)
                    p_base1, p_base2 = sorted_by_y[0], sorted_by_y[1]
                    if p_base1[0] > p_base2[0]:
                        p_base1, p_base2 = p_base2, p_base1
                    hit1, _ = world_map.project_image_point_to_world(p_base1[0], p_base1[1])
                    hit2, _ = world_map.project_image_point_to_world(p_base2[0], p_base2[1])
                    if hit1 is not None and hit2 is not None:
                        dx_world = float(hit2[0][0] - hit1[0][0])
                        dy_world = float(hit2[1][0] - hit1[1][0])
                        world_yaw = normalize_axis_angle(self.robot.pose.theta + math.atan2(dy_world, dx_world))

                face_label = obs.face_label if obs.face_label else "0-0"
                obj_id = f"domino_{face_label}"

                if obj_id in world_map.objects:
                    domino_obj = world_map.objects[obj_id]
                    domino_obj.pose.x = x_mm
                    domino_obj.pose.y = y_mm
                    domino_obj.pose.theta = world_yaw
                    domino_obj.is_fallen = obs.is_fallen
                else:
                    domino_obj = DominoObj(
                        id=obj_id,
                        x=x_mm,
                        y=y_mm,
                        z=(4.0 if obs.is_fallen else 12.0),
                        theta=world_yaw,
                        face_label=face_label,
                        face_confidence=obs.face_confidence,
                        is_fallen=obs.is_fallen,
                    )
                    world_map.objects[obj_id] = domino_obj

                domino_obj.is_visible = True
                domino_obj.is_missing = False
                domino_obj.confidence = obs.confidence
                domino_obj.image_center = obs.center_xy
                domino_obj.image_quad = obs.quad
                updated_ids.append(obj_id)

            if hasattr(world_map, "dispatch_update"):
                world_map.dispatch_update()

        return updated_ids

    # ------------------------------------------------------------------
    # FSM nodes & Parent Node (KnockDomino)
    # ------------------------------------------------------------------
    class FindStandingDomino(StateNode):
        def start(self, event=None):
            super().start(event)
            self.parent.target = None
            self.parent.target_id = None
            self.parent.knock_attempts = 0

            detector = getattr(self.robot, "domino_detector", None)
            if detector is None or self.parent._last_image is None:
                self.post_failure()
                return

            observations = detector.latest_observations()
            ids = self.parent._register_observations(observations)

            standing_ids = [oid for oid in ids if not self.robot.world_map.objects[oid].is_fallen]
            if not standing_ids:
                self.post_failure()
                return

            target_id = standing_ids[0]
            self.parent.target = self.robot.world_map.objects[target_id]
            self.parent.target_id = target_id
            t = self.parent.target
            print(f"[KNOCK] Targeting standing domino {target_id} at ({t.pose.x:.1f}, {t.pose.y:.1f}) mm")
            self.post_success()

    class ApproachDomino(PilotToPose):
        def start(self, event=None):
            target = self.parent.target
            if target is not None:
                self.target_object = target
                robot_pose = self.robot.pose
                dx = target.pose.x - robot_pose.x
                dy = target.pose.y - robot_pose.y
                dist = math.hypot(dx, dy) or 1e-3
                ux, uy = dx / dist, dy / dist
                self.target_pose = Pose(
                    target.pose.x - ux * APPROACH_STANDOFF_MM,
                    target.pose.y - uy * APPROACH_STANDOFF_MM,
                    theta=math.atan2(dy, dx),
                )
                print(f"[KNOCK] Approaching to {self.target_pose} ({APPROACH_STANDOFF_MM:.0f} mm standoff)")
            super().start(event)

    class KnockDomino(StateNode):
        """Parent node containing the complete swipe sequence"""
        class TurnKnock35(ActionNode):
            def start(self, event=None):
                super().start(event)
                print(f"[KNOCK] Turning {-KNOCK_TURN_DEG:.0f} deg...")
                self.robot.actuators["drive"].turn(self, math.radians(-KNOCK_TURN_DEG), None)

        class NudgeRightFast(ActionNode):
            def start(self, event=None):
                super().start(event)
                print(f"[KNOCK] Nudging {KNOCK_NUDGE_MM:.0f} mm right...")
                self.robot.actuators["drive"].sideways(self, KNOCK_NUDGE_MM, None)

        class NudgeLeftFast(ActionNode):
            def start(self, event=None):
                super().start(event)
                print(f"[KNOCK] Nudging {-KNOCK_NUDGE_MM:.0f} mm left...")
                self.robot.actuators["drive"].sideways(self, -KNOCK_NUDGE_MM, None)

        class TurnBackFromKnock(ActionNode):
            def start(self, event=None):
                super().start(event)
                print(f"[KNOCK] Turning back {KNOCK_TURN_DEG:.0f} deg...")
                self.robot.actuators["drive"].turn(self, math.radians(KNOCK_TURN_DEG), None)

        def setup(self):
            t35 = self.TurnKnock35().set_name("t35").set_parent(self)
            n_right = self.NudgeRightFast().set_name("n_right").set_parent(self)
            n_left = self.NudgeLeftFast().set_name("n_left").set_parent(self)
            t_back = self.TurnBackFromKnock().set_name("t_back").set_parent(self)

            CompletionTrans().add_sources(t35).add_destinations(n_right)
            CompletionTrans().add_sources(n_right).add_destinations(n_left)
            CompletionTrans().add_sources(n_left).add_destinations(t_back)
            
            return self

    class RetreatFromDomino(ActionNode):
        def start(self, event=None):
            super().start(event)
            print(f"[KNOCK] Backing away {KNOCK_RETREAT_MM:.0f} mm to check the result...")
            
            world_map = getattr(self.robot, "world_map", None)
            if world_map is not None and self.parent.target_id in world_map.objects:
                with world_map._lock if hasattr(world_map, "_lock") else nullcontext():
                    del world_map.objects[self.parent.target_id]
                    if hasattr(world_map, "dispatch_update"):
                        world_map.dispatch_update()
                print(f"[KNOCK] Cleared target {self.parent.target_id} from world map.")

            self.robot.actuators["drive"].forward(self, -KNOCK_RETREAT_MM, None)

    class CheckKnockResult(StateNode):
        def start(self, event=None):
            super().start(event)
            detector = getattr(self.robot, "domino_detector", None)
            if detector is None or self.parent._last_image is None or self.parent.target_id is None:
                self.post_failure()
                return

            observations = detector.detect(self.parent._last_image, frame_id=getattr(self.robot, "frame_count", None))
            if observations:
                self.parent._register_observations(observations)
            
            print(f"[KNOCK] World map updated post-retreat for target evaluation {self.parent.target_id}.")

            updated = self.robot.world_map.objects.get(self.parent.target_id)
            if updated is not None and updated.is_fallen:
                self.parent.target = updated
                self.post_success()
            else:
                self.post_failure()

    class ReportKnockSuccess(StateNode):
        def start(self, event=None):
            super().start(event)
            detector = getattr(self.robot, "domino_detector", None)
            if detector is not None:
                observations = detector.detect(self.parent._last_image, frame_id=getattr(self.robot, "frame_count", None))
                if observations:
                    self.parent._register_observations(observations)

            print(f"\nDomino knocked successfully! ({self.parent.target_id})")
            print(self.robot.world_map.objects.get(self.parent.target_id))
            self.parent.target = None
            self.parent.target_id = None
            self.post_completion()

    class RetryOrGiveUp(StateNode):
        def start(self, event=None):
            super().start(event)
            detector = getattr(self.robot, "domino_detector", None)
            if detector is not None:
                observations = detector.detect(self.parent._last_image, frame_id=getattr(self.robot, "frame_count", None))
                if observations:
                    self.parent._register_observations(observations)

            self.parent.knock_attempts += 1
            if self.parent.knock_attempts >= MAX_KNOCK_ATTEMPTS:
                print(f"[KNOCK] {self.parent.target_id} still standing after {self.parent.knock_attempts} attempts; giving up.")
                
                world_map = getattr(self.robot, "world_map", None)
                if world_map is not None and self.parent.target_id in world_map.objects:
                    with world_map._lock if hasattr(world_map, "_lock") else nullcontext():
                        del world_map.objects[self.parent.target_id]
                        if hasattr(world_map, "dispatch_update"):
                            world_map.dispatch_update()

                self.parent.target = None
                self.parent.target_id = None
                self.post_failure()
            else:
                print(f"[KNOCK] {self.parent.target_id} still standing (attempt {self.parent.knock_attempts}); retrying.")
                self.post_success()

    # ------------------------------------------------------------------
    # FSM wiring
    # ------------------------------------------------------------------
    def setup(self):
        intro = Print("\n[READY] Fast domino knock behavior active.").set_name("intro").set_parent(self)

        find = self.FindStandingDomino().set_name("find").set_parent(self)
        wait_scan = StateNode().set_name("wait_scan").set_parent(self)

        approach = self.ApproachDomino().set_name("approach").set_parent(self)
        approach_pilot_event = Print("[KNOCK] Pilot event during approach; rescanning...").set_name("approach_pilot_event").set_parent(self)

        knock_action = self.KnockDomino().set_name("knock_action").set_parent(self)
        retreat = self.RetreatFromDomino().set_name("retreat").set_parent(self)
        motion_failed = Print("[KNOCK] Motion action failed; rescanning...").set_name("motion_failed").set_parent(self)

        check_status = self.CheckKnockResult().set_name("check_status").set_parent(self)
        report = self.ReportKnockSuccess().set_name("report").set_parent(self)
        retry_decision = self.RetryOrGiveUp().set_name("retry_decision").set_parent(self)

        # --- intro / scan loop ---
        CompletionTrans().add_sources(intro).add_destinations(find)
        SuccessTrans().add_sources(find).add_destinations(approach)
        FailureTrans().add_sources(find).add_destinations(wait_scan)
        TimerTrans(RESCAN_DELAY_SEC).add_sources(wait_scan).add_destinations(find)

        # --- approach via the pilot ---
        CompletionTrans().add_sources(approach).add_destinations(knock_action)
        PilotTrans().add_sources(approach).add_destinations(approach_pilot_event)
        NullTrans().add_sources(approach_pilot_event).add_destinations(wait_scan)

        # --- knock action parent node & sequence flow ---
        CompletionTrans().add_sources(knock_action).add_destinations(retreat)
        FailureTrans().add_sources(knock_action).add_destinations(motion_failed)

        CompletionTrans().add_sources(retreat).add_destinations(check_status)
        FailureTrans().add_sources(retreat).add_destinations(motion_failed)

        NullTrans().add_sources(motion_failed).add_destinations(wait_scan)

        # --- check results ---
        SuccessTrans().add_sources(check_status).add_destinations(report)
        FailureTrans().add_sources(check_status).add_destinations(retry_decision)

        CompletionTrans().add_sources(report).add_destinations(wait_scan)

        SuccessTrans().add_sources(retry_decision).add_destinations(approach)
        FailureTrans().add_sources(retry_decision).add_destinations(wait_scan)

        return self