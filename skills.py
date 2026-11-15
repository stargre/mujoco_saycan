import numpy as np
import mujoco


class Skills:
    def __init__(self, ctx):
        self.c = ctx

    def _finger_contact(self, target_geom_id):
        """Return whether target geom is touching a finger geom, not the table."""
        model = self.c.rt.m
        data = self.c.rt.d
        contacts = []
        found = False
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
            })
            if "finger" in other_geom_name.lower() or "finger" in other_body_name.lower():
                found = True
        return found, contacts

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
            # These are hand-frame waypoints; the physical reach test validates IK.
            self.c.robot.move(object_position + np.array([0.0, 0.0, 0.15]), dur=1.5)
            self.c.robot.move(object_position + np.array([0.0, 0.0, 0.035]), dur=1.0)
            self.c.robot.grip(False, dur=0.8)
            self.c.robot.move(object_position + np.array([0.0, 0.0, 0.17]), dur=1.2)

            geom_id = mujoco.mj_name2id(
                self.c.rt.m, mujoco.mjtObj.mjOBJ_GEOM, target + "_geom"
            )
            finger_contact, contact_pairs = self._finger_contact(geom_id)
            if finger_contact:
                self.c.belief.held = target
                return self._result(
                    name, True, "grasp_contact_verified",
                    "Target remains in contact with a finger after the lift",
                    contact_pairs=contact_pairs,
                )

            return self._result(
                name, False, "grasp_not_verified",
                "No target-to-finger contact after lifting; table contact is not accepted as a grasp",
                contact_pairs=contact_pairs,
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
            # Place the held cube beside the reference cube, then release and retreat.
            release_hand_position = reference + np.array([0.075, 0.0, 0.055])
            self.c.robot.move(
                release_hand_position + np.array([0.0, 0.0, 0.13]), dur=1.5
            )
            self.c.robot.move(release_hand_position, dur=1.0)
            self.c.robot.grip(True, dur=0.8)
            self.c.robot.move(
                release_hand_position + np.array([0.0, 0.0, 0.15]), dur=1.2
            )

            # Let contact dynamics settle before evaluating the released object.
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
            finger_contact, contact_pairs = self._finger_contact(geom_id)
            # If the object is visible and no longer finger-held, update the belief.
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
                    finger_contact=finger_contact,
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
