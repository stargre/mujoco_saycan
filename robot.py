import mujoco
import numpy as np

JOINTS = [f"joint{i}" for i in range(1, 8)]
ACT = [f"actuator{i}" for i in range(1, 8)]
HOME = np.array([0., 0., 0., -1.57079, 0., 1.57079, -0.7853])

# Menagerie Panda MJCF:
# finger body offset along hand +Z = 0.0584 m
# main fingertip-pad geom offset inside finger = 0.0445 m
# The left/right pad Y offsets are mirrored, so their midpoint is on hand Z.
TCP_OFFSET_LOCAL = np.array([0.0, 0.0, 0.0584 + 0.0445])


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
        """World position of the Menagerie 'hand' body origin."""
        return self.d.xpos[self.ee].copy()

    def tcp_pos(self):
        """World position of the midpoint of the two main fingertip pads.

        The Menagerie Panda MJCF places each finger body 0.0584 m along the
        hand-local +Z axis and its main fingertip pad 0.0445 m farther along
        that axis. The mirrored finger Y offsets cancel at the midpoint.
        """
        rotation = self.d.xmat[self.ee].reshape(3, 3)
        return self.hand_pos() + rotation @ TCP_OFFSET_LOCAL

    def pos(self):
        """Compatibility alias: controlled end-effector position is the TCP."""
        return self.tcp_pos()

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

    def ik(self, x, max_iter=300, tol=0.003):
        """Position-only damped-least-squares IK for the fingertip TCP.

        IK temporarily changes qpos to evaluate forward kinematics and the
        Jacobian. It restores the live robot state before returning.
        """
        m = self.r.m
        target = np.asarray(x, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError(f"Target must be a finite 3D point, got {target}")

        q_start = self.q()
        solution = None
        last_error = float("inf")
        try:
            for _ in range(max_iter):
                mujoco.mj_forward(m, self.d)
                point = self.tcp_pos()
                error = target - point
                last_error = float(np.linalg.norm(error))
                if last_error < tol:
                    solution = self.q().copy()
                    break

                jacp = np.zeros((3, m.nv))
                jacr = np.zeros((3, m.nv))
                # Jacobian evaluated at the TCP point, attached to the hand body.
                mujoco.mj_jac(m, self.d, jacp, jacr, point, self.ee)
                jac = jacp[:, self.da]
                damping = 0.04
                delta = jac.T @ np.linalg.solve(
                    jac @ jac.T + damping**2 * np.eye(3), error
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
                raise RuntimeError(
                    f"TCP IK did not converge for target {target.tolist()}; "
                    f"last TCP error={last_error:.4f} m"
                )
            return solution
        finally:
            self.d.qpos[self.qa] = q_start
            mujoco.mj_forward(m, self.d)

    def move(self, x, dur=1.2):
        q_target = self.ik(x)
        return self.joints(q_target, dur)

    def grip(self, opened, dur=0.6):
        self.d.ctrl[self.g] = 255 if opened else 0
        for _ in range(max(1, int(dur / self.r.m.opt.timestep))):
            self.r.step()

    def home(self):
        self.grip(True)
        return self.joints(HOME, 2.0)
