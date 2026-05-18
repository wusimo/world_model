"""Action re-extraction from existing OXE tar shards.

For each episode in each shard, dump:
    /root/autodl-tmp/oxe_extracted/<dataset>/episode_NNNNNN/actions.npy   (T, 14)
    /root/autodl-tmp/oxe_extracted/<dataset>/episode_NNNNNN/states.npy    (T, 14)

Action schema per dataset is heterogeneous; we zero-pad to a 14-d slot layout:
    [world_vec(3), rot_delta(3), gripper(1), base_disp(2), base_rot(1), terminate(3), spare(1)]
This is a superset that fits jaco_play, bridge, kuka, fractal, taco_play. Per-dataset
mean/std normalization is left to compute_action_stats at training time.

States are extracted analogously from observation.* numeric fields.

Idempotent: skips episodes whose actions.npy already exists.
"""
from __future__ import annotations
import json, pickle, sys, time
from pathlib import Path
import numpy as np
import webdataset as wds

# 14-d action slot layout. -1 = absent in this dataset.
SLOT_MAP = {
    "jaco_play": {
        # world_vec(3) + gripper_closedness(1) + terminate(3) = 7d -> pad
        "world_vector":              [0, 1, 2],
        "gripper_closedness_action": [6],
        "terminate_episode":         [10, 11, 12],
    },
    "bridge": {
        "world_vector":  [0, 1, 2],
        "rotation_delta":[3, 4, 5],
        "open_gripper":  [6],
        "terminate_episode":[10, 11, 12],
    },
    "kuka": {
        "world_vector":  [0, 1, 2],
        "rotation_delta":[3, 4, 5],
        "gripper_closedness_action":[6],
        "terminate_episode":[10, 11, 12],
    },
    "fractal20220817_data": {
        "world_vector":  [0, 1, 2],
        "rotation_delta":[3, 4, 5],
        "gripper_closedness_action":[6],
        "base_displacement_vector":[7, 8],
        "base_displacement_vertical_rotation":[9],
        "terminate_episode":[10, 11, 12],
    },
    "taco_play": {
        # actions is already a 7d vector — splat into world_vec + rot_delta + gripper
        "actions": [0, 1, 2, 3, 4, 5, 6],
    },
}

def to_vec(action_dict, slot_map):
    out = np.zeros(14, dtype=np.float32)
    for key, slots in slot_map.items():
        if key in action_dict:
            v = action_dict[key]
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            n = min(len(slots), len(v))
            for i in range(n):
                out[slots[i]] = float(v[i])
    return out

def state_to_vec(obs_dict):
    # Concatenate up to 14 dims of numeric observation values (ee pose / joint pos)
    # for a unified state vector. Heuristic.
    out = np.zeros(14, dtype=np.float32)
    candidates = ["end_effector_cartesian_pos", "joint_pos", "base_pose_tool_reached",
                  "natural_language_embedding", "ee_pose"]
    i = 0
    for k in candidates:
        if i >= 14: break
        if k in obs_dict:
            v = np.asarray(obs_dict[k], dtype=np.float32).reshape(-1)
            n = min(14 - i, len(v))
            out[i:i+n] = v[:n]
            i += n
    return out

def main():
    IN = Path("/root/autodl-tmp/oxe")
    OUT = Path("/root/autodl-tmp/oxe_extracted")
    datasets = ["jaco_play", "taco_play", "kuka", "fractal20220817_data", "bridge"]
    for ds in datasets:
        slots = SLOT_MAP[ds]
        tars = sorted((IN / ds).glob("*.tar"))
        if not tars:
            print(f"{ds}: no tars in {IN/ds}, skipping")
            continue
        print(f"[{ds}] {len(tars)} shards")
        episode_offset = 0
        for tar in tars:
            t0 = time.time()
            n_eps = 0
            shard_ds = wds.WebDataset(str(tar), shardshuffle=False)
            for rec_idx, rec in enumerate(shard_ds):
                if "data.pickle" not in rec:
                    continue
                try:
                    ep = pickle.loads(rec["data.pickle"])
                except Exception:
                    continue
                steps = ep.get("steps", [])
                if not steps:
                    continue
                ep_idx = episode_offset + rec_idx
                ep_dir = OUT / ds / f"episode_{ep_idx:06d}"
                if not ep_dir.exists():
                    continue  # frames weren't extracted (e.g., too few frames) — skip
                actions_path = ep_dir / "actions.npy"
                if actions_path.exists():
                    n_eps += 1
                    continue
                actions = np.stack([to_vec(s.get("action", {}), slots) for s in steps], axis=0)
                states = np.stack([state_to_vec(s.get("observation", {})) for s in steps], axis=0)
                np.save(actions_path, actions)
                np.save(ep_dir / "states.npy", states)
                n_eps += 1
            episode_offset += rec_idx + 1
            print(f"  {tar.name}: {n_eps} episodes in {time.time()-t0:.1f}s")
    print("DONE")

if __name__ == "__main__":
    main()
