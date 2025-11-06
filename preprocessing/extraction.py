import cv2 as cv
import numpy as np
import json
import os

BODY_PARTS = ["Neck", "RShoulder", "RElbow", "RWrist",
              "LShoulder", "LElbow", "LWrist",
              "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle"]

def extract_keypoints(video_path, model_path="graph_opt.pb", threshold=0.2):
    net = cv.dnn.readNetFromTensorflow(model_path)
    cap = cv.VideoCapture(video_path)

    keypoints_sequence = []

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        h, w = frame.shape[:2]
        inp = cv.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5,127.5,127.5), swapRB=True)
        net.setInput(inp)
        out = net.forward()
        out = out[:, :19, :, :]

        frame_keypoints = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (w * point[0]) / out.shape[3]
            y = (h * point[1]) / out.shape[2]
            if conf > threshold:
                frame_keypoints.append([float(x), float(y)])
            else:
                frame_keypoints.append([None, None])

        keypoints_sequence.append(frame_keypoints)

    cap.release()
    return keypoints_sequence


if __name__ == "__main__":
    video = "dance.mp4"
    output_file = "pose_raw.json"

    data = extract_keypoints(video)
    with open(output_file, "w") as f:
        json.dump(data, f)

    print(f"Saved keypoints to {output_file}")