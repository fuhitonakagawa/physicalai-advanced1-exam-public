"""Genesisで時間発展するアームとFEM弾性体のラーメン具材。"""

import xml.etree.ElementTree as ET
from typing import Any, cast

import numpy as np
from defusedxml.ElementTree import fromstring

from src.config.settings import ROOT, Settings
from src.environment.backend import initialize
from src.environment.contact import bilateral_proximity
from src.environment.geometry import grasp_position, rotation_matrix
from src.environment.layout import food_layout
from src.environment.scene import INITIAL_ARM_POSITION, scene_xml
from src.environment.soft_geometry import (
    Points,
    grasp_geometry,
    served,
    shape_change,
    vector,
)
from src.experiment.images import encode_image
from src.experiment.messages import (
    Command,
    FoodState,
    ObservationMessage,
    Quaternion,
    Vector3,
)
from src.experiment.observations import quaternion_to_axis_angle

JOINT_NAMES = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]


def entity_xml(kind: str) -> ET.Element:
    """ロボット・固定物・具材を独立したMJCFに分ける。"""
    root = cast(ET.Element, fromstring(scene_xml()))
    world = root.find("worldbody")
    if world is None:
        raise ValueError("物理モデルのworldbodyがありません")
    for child in list(world):
        name = child.get("name", "")
        keep = (kind == "robot" and name == "link0") or (
            kind == "static"
            and name in ("table", "ramen_bowl", "egg_tray", "fish_cake_tray")
        )
        if not keep:
            world.remove(child)
    for tag in ("actuator", "tendon", "equality"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    if kind != "robot":
        for tag in ("contact", "asset"):
            element = root.find(tag)
            if element is not None:
                root.remove(element)
    return root


def vector3(tensor: Any) -> Vector3:
    """Genesisの3成分テンソルを通信可能な数値へ変換する。"""
    values = tensor.detach().cpu().numpy().reshape(3)
    return float(values[0]), float(values[1]), float(values[2])


def quaternion4(tensor: Any) -> Quaternion:
    """Genesisの姿勢を正規化したwxyzで返す。"""
    values = tensor.detach().cpu().numpy().reshape(4)
    values = values / np.linalg.norm(values)
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


class GenesisEnvironment:
    """物理演算、実カメラ、接触情報を提供する。"""

    def __init__(self, settings: Settings, cameras: bool) -> None:
        """Genesisバックエンドと物理場面を構築する。"""
        import genesis as gs

        self.gs = gs
        self.settings = settings
        self.steps = 0
        self.stable = {"egg": 0, "fish_cake": 0}
        initialize(gs, settings.genesis_backend, settings.seed)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.05, substeps=25),
            fem_options=gs.options.FEMOptions(
                use_implicit_solver=True,
                n_newton_iterations=2,
                n_pcg_iterations=40,
                n_linesearch_iterations=6,
                damping_alpha=1.0,
                damping_beta=settings.elastic_damping,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=True, friction_cone=gs.friction_cone.elliptic
            ),
            show_viewer=settings.show_viewer if cameras else False,
        )
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file=entity_xml("robot"), decimate=False),
            material=gs.materials.Rigid(
                gravity_compensation=1.0, coup_friction=3.0, coup_softness=0.001
            ),
        )
        self.scene.add_entity(
            gs.morphs.MJCF(file=entity_xml("static")),
            material=gs.materials.Rigid(coup_friction=3.0, coup_softness=0.001),
        )
        self.foods = {}
        for name, pose in food_layout(settings.seed).items():
            with np.load(
                ROOT / f"assets/soft_food/{name}.npz", allow_pickle=False
            ) as mesh:
                verts = mesh["vertices"]
                faces = mesh["faces"]
                volume = abs(
                    float(
                        np.einsum(
                            "ij,ij->i",
                            verts[faces[:, 0]],
                            np.cross(verts[faces[:, 1]], verts[faces[:, 2]]),
                        ).sum()
                        / 6
                    )
                )
            self.foods[name] = self.scene.add_entity(
                gs.morphs.Mesh(
                    file=str(ROOT / f"assets/soft_food/{name}.obj"),
                    pos=pose.position,
                    quat=pose.quaternion,
                    scale=1.0,
                    quality=False,
                ),
                material=gs.materials.FEM.Elastic(
                    E=settings.young_modulus_pa,
                    nu=settings.poisson_ratio,
                    rho=settings.soft_mass_kg / volume,
                    friction_mu=3.0,
                    model="linear_corotated",
                ),
            )
        self.hand = self.robot.get_link("hand")
        self.main_camera = (
            self.scene.add_camera(res=(256, 256), fov=45, GUI=False)
            if cameras
            else None
        )
        self.wrist_camera = (
            self.scene.add_camera(res=(256, 256), fov=75, GUI=False)
            if cameras
            else None
        )
        self.scene.build()
        self.rest = {name: self.soft_points(name)[0].copy() for name in self.foods}
        self.max_deformation = {name: 0.0 for name in self.foods}
        self.joints = [self.robot.get_joint(name) for name in JOINT_NAMES]
        self.dofs = [joint.dofs_idx_local[0] for joint in self.joints]
        self.qs = [joint.qs_idx_local[0] for joint in self.joints[:7]]
        self.robot.set_dofs_kp(
            [4500, 4500, 3500, 3500, 2000, 2000, 2000, 300, 300], self.dofs
        )
        self.robot.set_dofs_kv([450, 450, 350, 350, 200, 200, 200, 20, 20], self.dofs)
        self.robot.set_dofs_force_range(
            [-87] * 4 + [-12] * 3 + [-40] * 2, [87] * 4 + [12] * 3 + [40] * 2, self.dofs
        )
        initial = np.r_[INITIAL_ARM_POSITION, 0.04, 0.04]
        self.robot.set_dofs_position(initial, self.dofs)
        self.robot.control_dofs_position(initial, self.dofs)
        for _ in range(10):
            self.scene.step()
        self.configure_cameras()

    def configure_cameras(self) -> None:
        """外部カメラと手首カメラを物理リンクに合わせる。"""
        if self.main_camera is None or self.wrist_camera is None:
            return
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix(
            (0.6380177736, 0.3048497438, 0.3048498631, 0.6380177736)
        )
        transform[:3, 3] = [0.5886131747, 0, 1.490350024]
        self.main_camera.set_pose(transform=transform)
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix((0, 2**-0.5, 2**-0.5, 0))
        transform[:3, 3] = [0.05, 0, 0]
        self.wrist_camera.attach(self.hand, transform)

    def soft_points(self, name: str) -> tuple[Points, Points]:
        """FEMの実頂点位置と速度を返す。"""
        state = self.foods[name].get_state()
        return (
            state.pos.detach().cpu().numpy().reshape(-1, 3).astype(np.float64),
            state.vel.detach().cpu().numpy().reshape(-1, 3).astype(np.float64),
        )

    def food_state(self, name: str) -> FoodState:
        """頂点の近接と指の駆動力を、把持状態の目安として返す。"""
        points, velocities = self.soft_points(name)
        hand_pos = np.asarray(vector3(self.hand.get_pos()))
        hand_quat = quaternion4(self.hand.get_quat())
        local = (points - hand_pos) @ rotation_matrix(hand_quat)
        joints = self.robot.get_dofs_position(self.dofs).detach().cpu().numpy()
        forces = np.abs(
            self.robot.get_dofs_control_force(self.dofs).detach().cpu().numpy()
        )
        quaternion, width = grasp_geometry(points)
        return FoodState(
            position=vector(points.mean(axis=0)),
            speed=float(np.max(np.linalg.norm(velocities, axis=1))),
            finger_contacts=bilateral_proximity(local, joints[7], joints[8]),
            stable_steps=self.stable[name],
            bounds_min=vector(points.min(axis=0)),
            bounds_max=vector(points.max(axis=0)),
            grasp_quaternion=quaternion,
            half_width=width,
            left_load=float(forces[7]),
            right_load=float(forces[8]),
            deformation_rms=shape_change(points, self.rest[name]),
            vertex_count=len(points),
            contact_kind="vertex_proximity_and_drive_force",
        )

    def advance(self, command: Command) -> None:
        """目標を関節制御器へ渡し、物理状態を50ms進める。"""
        if self.steps >= self.settings.max_simulation_steps:
            raise RuntimeError("シミュレーションのステップ上限に達しました")
        qpos = (
            self.robot.inverse_kinematics(
                link=self.hand,
                pos=command.position,
                quat=command.quaternion,
                local_point=(0, 0, 0.097),
                dofs_idx_local=self.dofs[:7],
                max_samples=1,
                max_solver_iters=30,
            )
            .detach()
            .cpu()
            .numpy()
        )
        targets = np.r_[qpos[self.qs], command.opening, command.opening]
        self.robot.control_dofs_position(targets, self.dofs)
        self.scene.step()
        self.steps += 1
        hand = grasp_position(
            vector3(self.hand.get_pos()), quaternion4(self.hand.get_quat())
        )
        for name in self.foods:
            points, velocities = self.soft_points(name)
            if not np.isfinite(points).all() or not np.isfinite(velocities).all():
                raise RuntimeError("FEMの数値が不安定です")
            speed = float(np.max(np.linalg.norm(velocities, axis=1)))
            self.stable[name] = (
                self.stable[name] + 1 if served(points, speed, hand) else 0
            )
            self.max_deformation[name] = max(
                self.max_deformation[name], shape_change(points, self.rest[name])
            )

    def observation(self, request_id: str) -> ObservationMessage:
        """同じ物理時刻の画像・関節状態・具材状態を返す。"""
        quaternion = quaternion4(self.hand.get_quat())
        hand = grasp_position(vector3(self.hand.get_pos()), quaternion)
        axis_angle = quaternion_to_axis_angle(
            np.roll(np.asarray(quaternion, dtype=np.float32), -1)
        )
        joints = self.robot.get_dofs_position(self.dofs).detach().cpu().numpy()
        state = [*hand, *axis_angle.tolist(), float(joints[7]), -float(joints[8])]
        main_png = wrist_png = None
        if self.main_camera is not None and self.wrist_camera is not None:
            self.wrist_camera.move_to_attach()
            main_png = encode_image(
                np.asarray(self.main_camera.render()[0], dtype=np.uint8)
            )
            wrist_png = encode_image(
                np.asarray(self.wrist_camera.render()[0], dtype=np.uint8)
            )
        return ObservationMessage(
            request_id=request_id,
            steps=self.steps,
            state=state,
            hand_quaternion=quaternion,
            joint_positions=joints.tolist(),
            ingredients={name: self.food_state(name) for name in self.foods},
            main_png=main_png,
            wrist_png=wrist_png,
        )

    def close(self) -> None:
        """Genesisの物理・描画リソースを解放する。"""
        self.gs.destroy()
