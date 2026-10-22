# from preprocess.py
import os
import numpy as np

def main():
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_DIR, "data", "mixamo_joints3d", "joints3d_npy")
    OUT_DIR = os.path.join(PROJECT_DIR, "data", "motion")
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    len_frames = 64

    names = sorted(name for name in os.listdir(DATA_DIR) if name.endswith(".npy"))
    for name in names:
        # .npy files    
        stem = os.path.splitext(name)[0]
        out_path = os.path.join(OUT_DIR, stem)
        # skip if already processed
        if os.path.exists(out_path):
            continue

        anim_path = os.path.join(DATA_DIR, name)
        animation = np.load(anim_path)
        total_frames = animation.shape[2]
        nr_motions = total_frames // (len_frames // 2) - 1
        if nr_motions <= 0:
            continue

        os.makedirs(out_path)
        for i in range(nr_motions):
            save_path = os.path.join(out_path, '{}.npy'.format(i + 1))
            window = animation[:, :, i * (len_frames // 2): i * (len_frames // 2) + len_frames]
            np.save(save_path, window)

if __name__ == "__main__":
    main()
