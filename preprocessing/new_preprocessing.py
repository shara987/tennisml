import numpy as np

def resample_sequence(seq, target_frames=100):
    seq = np.array(seq)
    original_len = len(seq)
    idx = np.linspace(0, original_len - 1, target_frames).astype(int)
    return seq[idx]

def smooth(seq, alpha=0.2):
    smoothed = seq.copy()
    for t in range(1, len(seq)):
        smoothed[t] = alpha*seq[t] + (1-alpha)*smoothed[t-1]
    return smoothed

def preprocess_sequence(raw_seq, target_frames=100):
    # Replace None with zeros
    seq = np.array([[p if p != [None, None] else [0.0, 0.0]
                     for p in frame] for frame in raw_seq], dtype=float)

    # Smooth confidence noise
    seq = smooth(seq)

    # Compute global pelvis center
    pelvis = (seq[:, 7] + seq[:, 10]) / 2
    center = pelvis.mean(axis=0)
    seq -= center

    # Compute global shoulder width
    shoulders_valid = []
    for f in seq:
        RS, LS = f[1], f[4]
        if np.linalg.norm(RS) != 0 and np.linalg.norm(LS) != 0:
            shoulders_valid.append(np.linalg.norm(RS - LS))
    width = np.median(shoulders_valid) if len(shoulders_valid) > 0 else 1.0

    seq /= width

    # Optional stabilization
    seq = np.clip(seq, -5.0, 5.0)

    # Resample to fixed number of frames
    T = len(seq)
    idx = np.linspace(0, T-1, target_frames).astype(int)
    seq = seq[idx]

    return seq
