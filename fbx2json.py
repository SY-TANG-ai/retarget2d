# based on dataset/fbx2joints3d.py
import bpy
import json
import os
import time

import numpy as np
from mathutils import Vector

BASE_JOINT_NAMES = [
    "Head",
    "Neck",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "Hips",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "download_mixamo")
OUT_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "mixamo_joints3d")
MIN_NR_FRAMES = 64


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clear_scene_and_import_fbx(filepath):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=filepath)


def get_armature_object():
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    raise ValueError("No armature object found in imported FBX.")


def resolve_pose_bone(pose_bones, bone_name):
    bone = pose_bones.get(bone_name)
    if bone is not None:
        return bone

    suffix = ":" + bone_name
    for candidate in pose_bones:
        if candidate.name.endswith(suffix):
            return candidate

    raise KeyError("Bone '{}' not found in armature.".format(bone_name))


def get_frame_range(armature):
    animation_data = armature.animation_data
    if animation_data is not None:
        action = animation_data.action
        if action is not None:
            frame_start, frame_end = action.frame_range
            return int(frame_start), int(frame_end)

        for track in animation_data.nla_tracks:
            for strip in track.strips:
                return int(strip.frame_start), int(strip.frame_end)

    scene = bpy.context.scene
    if scene.frame_end > scene.frame_start:
        return int(scene.frame_start), int(scene.frame_end)

    raise ValueError("No animation frame range found for imported FBX.")


def get_joint3d_positions(armature, joint_names, frame_idx):
    bpy.context.scene.frame_set(frame_idx)
    pose_bones = armature.pose.bones

    out_dict = {"pose_keypoints_3d": []}
    for name in joint_names:
        # bone origin -> armature coordinate -> global space
        bone = resolve_pose_bone(pose_bones, name)
        global_location = armature.matrix_world @ bone.matrix @ Vector((0.0, 0.0, 0.0))
        out_dict["pose_keypoints_3d"].extend(
            [global_location[0], global_location[1], global_location[2]]
        )
    return out_dict

# fbx -> json
def main():
    ensure_dir(OUT_DATA_DIR)

    files = sorted(
        name for name in os.listdir(SRC_DATA_DIR) if name.lower().endswith(".fbx")
    )
    for j, name in enumerate(files, start=1):
        path = os.path.join(SRC_DATA_DIR, name)
        since = time.time()

        clear_scene_and_import_fbx(path)
        armature = get_armature_object()
        frame_start, frame_end = get_frame_range(armature)
        if frame_end - frame_start < MIN_NR_FRAMES - 1:
            continue

        save_dir = os.path.join(OUT_DATA_DIR, "jointsDict", name)
        ensure_dir(save_dir)

        for frame_idx in range(frame_start, frame_end + 1):
            out_dict = get_joint3d_positions(armature, BASE_JOINT_NAMES, frame_idx)
            save_path = os.path.join(save_dir, "{:04d}_keypoints.json".format(frame_idx))
            with open(save_path, "w") as f:
                json.dump(out_dict, f)

        process_time = time.time() - since
        print("[{}/{}] ({:.2f}s) {}".format(j, len(files), process_time, path))


def json2npy():
    data_root = os.path.join(OUT_DATA_DIR, "jointsDict")
    npy_dir = os.path.join(OUT_DATA_DIR, "joints3d_npy")
    ensure_dir(npy_dir)

    names = sorted(
        name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))
    )
    for name in names:
        joint_dir = os.path.join(data_root, name)
        frame_files = sorted(
            file_name
            for file_name in os.listdir(joint_dir)
            if file_name.endswith("_keypoints.json")
        )

        motion = []
        for frame_file in frame_files:
            with open(os.path.join(joint_dir, frame_file)) as f:
                info = json.load(f)
            joint = np.array(info["pose_keypoints_3d"], dtype=np.float32).reshape((-1, 3))
            motion.append(joint[:15, :])

        if not motion:
            continue

        motion = np.stack(motion, axis=2)
        stem = os.path.splitext(name)[0]
        save_path = os.path.join(npy_dir, "{}.npy".format(stem))
        np.save(save_path, motion)


if __name__ == "__main__":
    main()
    json2npy()
