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

    def _robot_table_contacts(self):
        """Return robot/table contacts, excluding the two movable cubes."""
        model = self.c.rt.m
        data = self.c.rt.d
        table_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "table"
        )
        if table_id < 0:
            return []
        found = []
        for index in range(data.ncon):
            contact = data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if table_id not in (g1, g2):
                continue
            other = g2 if g1 == table_id else g1
            body_id = int(model.geom_bodyid[other])
            body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            )
            if body_name in ("red_cube", "blue_cube"):
                continue
            found.append({
                "body": body_name,
                "distance_m": float(contact.dist),
                "position_world": np.asarray(contact.pos).round(6).tolist(),
            })
        return found

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
            # Empirical TCP-to-cube-center offset from the validated grasp
            # in this fixed top-down scene. This is a scene calibration, not
            # a runtime ground-truth object-pose lookup. Recalibrate if the
            # gripper orientation, cube geometry, or grasp strategy changes.
            self.c.held_tcp_to_object_offset_world = np.array(
                [0.00015, -0.00728, 0.01255], dtype=float
            )
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

            held_offset = getattr(self.c, "held_tcp_to_object_offset_world", None)
            if held_offset is None:
                return self._result(
                    name, False, "grasp_transform_unavailable",
                    "No calibrated TCP-to-object transform is available for this grasp",
                )
            held_offset = np.asarray(held_offset, dtype=float)
            if held_offset.shape != (3,) or not np.all(np.isfinite(held_offset)):
                return self._result(name, False, "invalid_grasp_transform", "Invalid held-object transform")

            model = self.c.rt.m
            data = self.c.rt.d
            geom_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, object_name + "_geom"
            )
            table_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, "table"
            )
            if geom_id < 0 or table_id < 0:
                return self._result(name, False, "missing_scene_geometry", "Object or table geom is missing")

            reference = np.asarray(
                self.c.belief.objects[target_name]["position"], dtype=float
            )
            # This scene uses a fixed top-down grasp and world-X placement.
            # The target position comes from RGB-D belief; only the known table
            # and cube geometry are used to set a safe release height.
            table_top = float(model.geom_pos[table_id][2] + model.geom_size[table_id][2])
            object_half_height = float(model.geom_size[geom_id][2])
            goal_object = reference.copy()
            goal_object[0] += 0.075
            goal_object[2] = table_top + object_half_height + 0.003
            goal_tcp = goal_object + held_offset
            preplace_tcp = goal_tcp + np.array([0.0, 0.0, 0.12])
            release_rotation = self.c.robot.rotation()

            required_fingers = {"left_finger", "right_finger"}
            initial_fingers, initial_contacts = self._finger_contact(geom_id)
            if not required_fingers.issubset(initial_fingers):
                return self._result(
                    name, False, "grasp_contact_lost_before_place",
                    "Both fingers must still contact the object before transport",
                    contacting_fingers=sorted(initial_fingers),
                    contact_pairs=initial_contacts,
                )

            self._stage("before_preplace")
            self.c.robot.move(
                preplace_tcp, dur=2.0, target_rotation=release_rotation
            )
            robot_table_contacts = self._robot_table_contacts()
            if robot_table_contacts:
                return self._result(
                    name, False, "robot_table_collision",
                    "Robot-table contact detected while moving to pre-place",
                    contacts=robot_table_contacts,
                )
            self._stage("preplace_reached")

            self.c.robot.move(
                goal_tcp, dur=1.3, target_rotation=release_rotation
            )
            robot_table_contacts = self._robot_table_contacts()
            if robot_table_contacts:
                return self._result(
                    name, False, "robot_table_collision",
                    "Robot-table contact detected during placement approach",
                    contacts=robot_table_contacts,
                )
            self._stage("place_approach_reached")

            approach_fingers, approach_contacts = self._finger_contact(geom_id)
            if not required_fingers.issubset(approach_fingers):
                return self._result(
                    name, False, "grasp_lost_during_transport",
                    "Bilateral finger contact was lost before release",
                    contacting_fingers=sorted(approach_fingers),
                    contact_pairs=approach_contacts,
                )

            self.c.robot.grip(True, dur=0.8)
            self._stage("gripper_opened")
            for _ in range(max(1, int(0.7 / model.opt.timestep))):
                self.c.rt.step()

            self.c.robot.move(
                goal_tcp + np.array([0.0, 0.0, 0.15]),
                dur=1.3, target_rotation=release_rotation,
            )
            robot_table_contacts = self._robot_table_contacts()
            self._stage("retreated")

            for _ in range(max(1, int(0.5 / model.opt.timestep))):
                self.c.rt.step()
            self.c.observe()

            finger_bodies, contact_pairs = self._finger_contact(geom_id)
            finger_contact = bool(finger_bodies)
            table_contact = self._object_table_contact(geom_id)
            if not finger_contact:
                self.c.belief.held = None

            if object_name not in self.c.belief.objects:
                return self._result(
                    name, False, "placement_unobserved",
                    "Released object is not visible after retreat",
                    finger_contact=finger_contact,
                    table_contact=table_contact,
                    contact_pairs=contact_pairs,
                )

            placed = np.asarray(
                self.c.belief.objects[object_name]["position"], dtype=float
            )
            reference_after = np.asarray(
                self.c.belief.objects[target_name]["position"], dtype=float
            )
            distance_xy = float(np.linalg.norm(placed[:2] - reference_after[:2]))
            goal_xy_error = float(np.linalg.norm(placed[:2] - goal_object[:2]))
            height_error = float(abs(placed[2] - goal_object[2]))

            verified = (
                not finger_contact
                and table_contact
                and goal_xy_error <= 0.045
                and 0.05 <= distance_xy <= 0.13
                and height_error <= 0.04
                and not robot_table_contacts
            )
            self._stage("placement_checked")
            if not verified:
                return self._result(
                    name, False, "placement_not_verified",
                    "Release completed, but one or more physical/perceptual postconditions failed",
                    distance_xy=distance_xy,
                    goal_xy_error=goal_xy_error,
                    height_error=height_error,
                    finger_contact=finger_contact,
                    contacting_fingers=sorted(finger_bodies),
                    table_contact=table_contact,
                    robot_table_contacts=robot_table_contacts,
                    placed_position=placed.tolist(),
                    reference_position=reference_after.tolist(),
                    goal_object_position=goal_object.tolist(),
                    contact_pairs=contact_pairs,
                )

            if "place" not in self.c.belief.completed:
                self.c.belief.completed.append("place")
            self._stage("placement_verified")
            return self._result(
                name, True, "placement_verified",
                "Object is on the table, no longer finger-held, and within the target placement tolerance",
                distance_xy=distance_xy,
                goal_xy_error=goal_xy_error,
                height_error=height_error,
                finger_contact=finger_contact,
                table_contact=table_contact,
                placed_position=placed.tolist(),
                reference_position=reference_after.tolist(),
                goal_object_position=goal_object.tolist(),
                grasp_transform_source="scene_calibrated_tcp_to_object_offset",
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
