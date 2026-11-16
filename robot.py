import mujoco
import numpy as np

JOINTS = [f"joint{i}" for i in range(1, 8)]
ACT = [f"actuator{i}" for i in range(1, 8)]
HOME = np.array([0., 0., 0., -1.57079, 0., 1.57079, -0.7853])

# Menagerie Panda: hand -> finger body (58.4 mm) -> main pad center (44.5 mm).
# The mirrored finger-local Y offsets cancel when taking the two pad centers'
# midpoint, leaving this fixed offset in the hand frame.
TCP_OFFSET_LOCAL = np.array([0.0, 0.0, 0.1029])


class Panda:
    def __init__(self, rt):
        self.r = rt
        m = rt.m
        self.d = rt.d
        self.qa = [
            m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINTS
        ]
        self.da = [
            m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in JOINTS
        ]
        self.ai = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ACT
        ]
        self.g = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"
        )
        self.ee = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )

    def q(self):
        return self.d.qpos[self.qa].copy()

    def hand_pos(self):
        return self.d.xpos[self.ee].copy()

    def rotation(self):
        """World-from-hand rotation matrix; TCP axes are aligned with hand."""
        return self.d.xmat[self.ee].reshape(3, 3).copy()

    def tcp_pos(self):
        """World position of the midpoint of the two main fingertip pads."""
        return self.hand_pos() + self.rotation() @ TCP_OFFSET_LOCAL

    def pos(self):
        """Compatibility alias: controlled end-effector position is the TCP."""
        return self.tcp_pos()

    @staticmethod
    def _rotation_error(current, target):
        """World-frame SO(3) logarithm, expressed as a 3D rotation vector."""
        relative = target @ current.T
        cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        angle = float(np.arccos(cosine))
        vee = np.array([
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ])
        if angle < 1e-7:
            return 0.5 * vee
        if np.pi - angle < 1e-5:
            # The grasp controller starts near its target orientation. This
            # branch keeps the log map finite if a caller supplies a near-pi
            # rotation difference.
            eigvals, eigvecs = np.linalg.eig(relative)
            axis = np.real(eigvecs[:, int(np.argmin(np.abs(eigvals - 1.0)))])
            axis /= max(float(np.linalg.norm(axis)), 1e-12)
            return angle * axis
        return (angle / (2.0 * np.sin(angle))) * vee

    def joints(self, target, dur=1.3):
        target = np.asarray(target, dtype=float)
        if target.shape != (7,):
            raise ValueError(f"Expected 7 joint targets, got shape {target.shape}")
        start = self.q()
        n = max(2, int(dur / self.r.m.opt.timestep))
        for i in range(n):
            t = (i + 1) / n
            s = 10 * t**3 - 15 * t**4 + 6 * t**5
            self.d.ctrl[self.ai] = start + s * (target - start)
            self.r.step()
        return self.q()

    def ik(
        self,
        x,
        max_iter=400,
        tol=0.003,
        target_rotation=None,
        orientation_tol=0.015,
    ):
        """Damped-least-squares IK for TCP position and optional hand orientation.

        The solver temporarily changes qpos to evaluate forward kinematics and
        Jacobians. It always restores the live joint state before returning.
        When target_rotation is supplied, a 6D position/orientation task is
        solved; otherwise this remains the original position-only 3D IK.
        """
        m = self.r.m
        target = np.asarray(x, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError(f"Target must be a finite 3D point, got {target}")

        if target_rotation is not None:
            target_rotation = np.asarray(target_rotation, dtype=float)
            if target_rotation.shape != (3, 3) or not np.all(np.isfinite(target_rotation)):
                raise ValueError("target_rotation must be a finite 3x3 matrix")
            if not np.allclose(target_rotation.T @ target_rotation, np.eye(3), atol=1e-3):
                raise ValueError("target_rotation must be orthonormal")
            if np.linalg.det(target_rotation) < 0.999:
                raise ValueError("target_rotation must be a proper rotation matrix")

        q_start = self.q()
        solution = None
        last_pos_error = float("inf")
        last_rot_error = float("inf")
        try:
            for _ in range(max_iter):
                mujoco.mj_forward(m, self.d)
                pos_error = target - self.tcp_pos()
                last_pos_error = float(np.linalg.norm(pos_error))

                jacp = np.zeros((3, m.nv))
                jacr = np.zeros((3, m.nv))
                mujoco.mj_jac(m, self.d, jacp, jacr, self.tcp_pos(), self.ee)
                jacp = jacp[:, self.da]

                if target_rotation is None:
                    if last_pos_error < tol:
                        solution = self.q().copy()
                        break
                    jac = jacp
                    task_error = pos_error
                else:
                    rot_error = self._rotation_error(self.rotation(), target_rotation)
                    last_rot_error = float(np.linalg.norm(rot_error))
                    if last_pos_error < tol and last_rot_error < orientation_tol:
                        solution = self.q().copy()
                        break
                    orientation_weight = 0.15
                    jac = np.vstack((
                        jacp,
                        orientation_weight * jacr[:, self.da],
                    ))
                    task_error = np.concatenate((
                        pos_error,
                        orientation_weight * rot_error,
                    ))

                damping = 0.05
                delta = jac.T @ np.linalg.solve(
                    jac @ jac.T + damping**2 * np.eye(jac.shape[0]),
                    task_error,
                )
                q_next = self.q() + np.clip(delta, -0.05, 0.05)

                for j, name in enumerate(JOINTS):
                    jid = mujoco.mj_name2id(
                        m, mujoco.mjtObj.mjOBJ_JOINT, name
                    )
                    low, high = m.jnt_range[jid]
                    q_next[j] = np.clip(q_next[j], low + 0.01, high - 0.01)

                self.d.qpos[self.qa] = q_next

            if solution is None:
                detail = f"position error={last_pos_error:.4f} m"
                if target_rotation is not None:
                    detail += f", orientation error={last_rot_error:.4f} rad"
                raise RuntimeError(
                    f"TCP IK did not converge for target {target.tolist()}; {detail}"
                )
            return solution
        finally:
            self.d.qpos[self.qa] = q_start
            mujoco.mj_forward(m, self.d)

    def ik_pose(self, position, rotation, **kwargs):
        """Solve a full TCP pose while preserving the requested orientation."""
        return self.ik(position, target_rotation=rotation, **kwargs)

    def move(self, x, dur=1.2, target_rotation=None):
        if target_rotation is None:
            q_target = self.ik(x)
        else:
            q_target = self.ik_pose(x, target_rotation)
        return self.joints(q_target, dur)

    def grip(self, opened, dur=0.6):
        self.d.ctrl[self.g] = 255 if opened else 0
        for _ in range(max(1, int(dur / self.r.m.opt.timestep))):
            self.r.step()

    def home(self):
        self.grip(True)
        return self.joints(HOME, 2.0)
