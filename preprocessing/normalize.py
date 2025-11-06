import json
import numpy as np

#returns midpoint between 2 pts
def midpoint(p1, p2):
    return (np.array(p1) + np.array(p2)) / 2

#normalizes each frame for easier comparison
def normalize_pose(keypoints_sequence):
    normalized = []

    for frame in keypoints_sequence:
        frame = np.array(frame, dtype=float)

        #if the entire frame is missing skip 
        if np.isnan(frame).all():
            normalized.append(frame.tolist())
            continue

        #finding the 'center of the body' using l/rhip
        RHip = frame[7]
        LHip = frame[10]
        #skip if missing
        if np.isnan(RHip).any() or np.isnan(LHip).any():
            normalized.append(frame.tolist())
            continue

        center = midpoint(RHip, LHip)
        frame -= center

        #normalizes size by dividing coords by shoulder width
        RShoulder = frame[1]
        LShoulder = frame[4]
        if np.isnan(RShoulder).any() or np.isnan(LShoulder).any():
            normalized.append(frame.tolist())
            continue

        shoulder_width = np.linalg.norm(RShoulder - LShoulder)
        #preventing div by 0
        if shoulder_width > 0:
            frame /= shoulder_width

        normalized.append(frame.tolist())

    return normalized

if __name__ == "__main__":
    with open("pose_raw.json", "r") as f:
        raw = json.load(f)

    normalized = normalize_pose(raw)

    with open("pose_normalized.json", "w") as f:
        json.dump(normalized, f)