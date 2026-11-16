import numpy as np
import mujoco


class Skills:
    def __init__(self, ctx):
        self.c = ctx

    def _stage(self, name):
        callback = getattr(self.c, "on_stage", None)
        if callback is not None:
            callback(name)

    def _finger_contact(self, target_geom_id):
        """Return contacting finger body names and detailed target contact pairs."""
        model = self.c.rt.m
        data = self.c.rt.d
        finger_bodies = set()
        contacts = []
        for index in range(data.ncon):
            contact = data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if target_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == target_geom_id else g1
            other_geom_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            )
            other_body = int(model.geom_bodyid[other])
            other_body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body) or ""
            )
            contacts.append({
                "other_geom": other_geom_name,
                "other_body": other_body_name,
                "distance_m": float(contact.dist),
                "position_world": np.asarray(contact.pos).round(6).tolist(),
            })
            if other_body_name in ("left_finger", "right_finger"):
                finger_bodies.add(other_body_name)
        return finger_bodies, contacts

    def _object_table_contact(self, target_geom_id):
        model = self.c.rt.m
        data = self.c.rt.d
        table_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "table"
        )
        return any(
            {int(data.contact[i].geom1), int(data.contact[i].geom2)}
            == {int(table_id), int(target_geom_id)}
            for i in range(data.ncon)
        )

    def execute(self, call):
        name = call["name"]
        args = call["args"]
        target = args.get("target")

        if name in ("Explore", "Find"):
            self.c.observe()
            ok = name == "Explore" or target in self.c.belief.objects
            return self._result(
                name, ok,
                "skill_complete" if ok else "target_not_found",
                "RGB-D observation refreshed",
            )

        if name == "PickUp":
            self.c.observe()
            if self.c.belief.held is not None:
                return self._result(name, False, "gripper_not_empty", "Belief says gripper is occupied")
            if target not in self.c.belief.objects:
                return self._result(name, False, "target_not_found", "Target is not visible")

            object_position = np.asarray(
                self.c.belief.objects[target]["position"], dtype=float
            )
            geom_id = mujoco.mj_name2id(
                self.c.rt.m, mujoco.mjtObj.mjOBJ_GEOM, target + "_geom"
            )
            body_id = mujoco.mj_name2id(
                self.c.rt.m, mujoco.mjtObj.mjOBJ_BODY, target
            )
            if geom_id < 0 or body_id < 0:
                return self._result(name, False, "invalid_target", f"Missing MuJoCo body/geom for {target}")

            # Lock the current home grasp orientation: the two fingers close
            # along the world-X direction in this tabletop scene.
            grasp_rotation = self.c.robot.rotation()
            pregrasp = object_position + np.array([0.0, 0.0, 0.10])
            grasp_tcp = object_position.copy()
            initial_object_z = float(self.c.rt.d.xpos[body_id][2])

            self._stage("before_pregrasp")
            self.c.robot.move(pregrasp, dur=1.8, target_rotation=grasp_rotation)
            self._stage("pregrasp_reached")

            # TCP is the midpoint of the main fingertip pads. Placing this
            # point at the perceived cube center aligns the pads with the
            # upper side faces; the previous +35 mm offset left them above it.
            self.c.robot.move(grasp_tcp, dur=1.2, target_rotation=grasp_rotation)
            self._stage("grasp_approach_reached")

            self.c.robot.grip(False, dur=1.0)
            for _ in range(max(1, int(0.25 / self.c.rt.m.opt.timestep))):
                self.c.rt.step()
            finger_bodies, contact_pairs = self._finger_contact(geom_id)
            self._stage("gripper_closed")

            required_fingers = {"left_finger", "right_finger"}
            if not required_fingers.issubset(finger_bodies):
                self.c.robot.grip(True, dur=0.4)
                self.c.robot.move(pregrasp, dur=1.0, target_rotation=grasp_rotation)
                return self._result(
                    name, False, "bilateral_grasp_not_verified",
                    "Both finger bodies must contact the target before lifting",
                    contacting_fingers=sorted(finger_bodies),
                    contact_pairs=contact_pairs,
                )

            lift_tcp = grasp_tcp + np.array([0.0, 0.0, 0.12])
            self.c.robot.move(lift_tcp, dur=1.5, target_rotation=grasp_rotation)
            for _ in range(max(1, int(0.35 / self.c.rt.m.opt.timestep))):
                self.c.rt.step()

            lifted_fingers, lift_contacts = self._finger_contact(geom_id)
            final_object_z = float(self.c.rt.d.xpos[body_id][2])
            lift_delta = final_object_z - initial_object_z
            object_on_table = self._object_table_contact(geom_id)
            self._stage("lift_completed")

            # MuJoCo object pose is used here only as a simulator-side test
            # oracle. Planner candidates still receive camera belief and
            # contact-derived held state, not ground-truth object coordinates.
            lift_verified = (
                required_fingers.issubset(lifted_fingers)
                and lift_delta >= 0.06
                and not object_on_table
            )
            if not lift_verified:
                self.c.belief.held = None
                return self._result(
                    name, False, "lift_not_verified",
                    "The cube did not remain bilaterally grasped and clear the table",
                    contacting_fingers=sorted(lifted_fingers),
                    contact_pairs=lift_contacts,
                    simulator_object_lift_m=lift_delta,
                    object_still_on_table=object_on_table,
                )

            self.c.belief.held = target
            self.c.belief.completed = [
                item for item in self.c.belief.completed if item != "pick"
            ]
            self.c.belief.completed.append("pick")
            self.c.observe()
            self._stage("grasp_verified")
            return self._result(
                name, True, "grasp_and_lift_verified",
                "Both fingers remain in contact and the cube is lifted clear of the table",
                contacting_fingers=sorted(lifted_fingers),
                simulator_object_lift_m=lift_delta,
                object_still_on_table=object_on_table,
                contact_pairs=lift_contacts,
            )

        if name == "PlaceNear":
            object_name = args["object"]
            target_name = args["target"]
            if self.c.belief.held != object_name:
                return self._result(name, False, "not_holding_object", "Requested object is not held")
            if target_name not in self.c.belief.objects:
                return self._result(name, False, "target_not_found", "Placement reference is not visible")

            reference = np.asarray(
                self.c.belief.objects[target_name]["position"], dtype=float
            )
            release_hand_position = reference + np.array([0.075, 0.0, 0.055])
            release_rotation = self.c.robot.rotation()
            self.c.robot.move(
                release_hand_position + np.array([0.0, 0.0, 0.13]),
                dur=1.5, target_rotation=release_rotation,
            )
            self.c.robot.move(
                release_hand_position, dur=1.0, target_rotation=release_rotation,
            )
            self.c.robot.grip(True, dur=0.8)
            self.c.robot.move(
                release_hand_position + np.array([0.0, 0.0, 0.15]),
                dur=1.2, target_rotation=release_rotation,
            )

            for _ in range(max(1, int(0.4 / self.c.rt.m.opt.timestep))):
                self.c.rt.step()
            self.c.observe()

            if object_name not in self.c.belief.objects:
                return self._result(
                    name, False, "placement_unobserved",
                    "Released object is not visible after retreat; placement cannot be verified",
                )

            placed = np.asarray(
                self.c.belief.objects[object_name]["position"], dtype=float
            )
            reference = np.asarray(
                self.c.belief.objects[target_name]["position"], dtype=float
            )
            distance_xy = float(np.linalg.norm(placed[:2] - reference[:2]))
            height_difference = float(abs(placed[2] - reference[2]))

            geom_id = mujoco.mj_name2id(
                self.c.rt.m, mujoco.mjtObj.mjOBJ_GEOM, object_name + "_geom"
            )
            finger_bodies, contact_pairs = self._finger_contact(geom_id)
            finger_contact = bool(finger_bodies)
            if not finger_contact:
                self.c.belief.held = None

            verified = (
                not finger_contact
                and distance_xy <= 0.15
                and height_difference <= 0.08
            )
            if not verified:
                return self._result(
                    name, False, "placement_not_verified",
                    "Release command ran, but camera-based postcondition failed",
                    distance_xy=distance_xy,
                    height_difference=height_difference,
                    contacting_fingers=sorted(finger_bodies),
                    contact_pairs=contact_pairs,
                    placed_position=placed.tolist(),
                    reference_position=reference.tolist(),
                )

            self.c.belief.completed.append("place")
            return self._result(
                name, True, "placement_verified",
                "Object is visible near the reference object and no longer finger-held",
                distance_xy=distance_xy,
                height_difference=height_difference,
            )

        if name == "Done":
            if self.c.task_place:
                ok = (
                    self.c.belief.held is None
                    and "place" in self.c.belief.completed
                )
            else:
                ok = self.c.belief.held == getattr(
                    self.c, "task_target", None
                )
            return self._result(
                name, ok,
                "task_complete" if ok else "task_not_complete",
                "Task postcondition checked against belief state",
            )

        if name == "NavigateTo":
            position = np.asarray(
                self.c.belief.objects[target]["position"], dtype=float
            )
            self.c.robot.move(position + np.array([0.0, 0.0, 0.18]))
            return self._result(name, True, "skill_complete", "Pre-grasp reached")

        if name == "Reach":
            position = np.asarray(
                self.c.belief.objects[target]["position"], dtype=float
            )
            self.c.robot.move(position + np.array([0.0, 0.0, 0.04]))
            return self._result(name, True, "skill_complete", "Approach reached")

        return self._result(name, False, "unknown_skill", name)

    @staticmethod
    def _result(name, success, status, reason, **extra):
        return {
            "skill": name,
            "success": success,
            "status": status,
            "reason": reason,
            **extra,
        }
