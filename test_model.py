import torch
import cv2 as cv
import numpy as np
from train_model import VAE
from preprocessing.new_preprocessing import preprocess_sequence 
from preprocessing.bodypart_extraction import extract_keypoints

BODY_PARTS = ["Neck", "RShoulder", "RElbow", "RWrist",
              "LShoulder", "LElbow", "LWrist",
              "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = VAE(latent_dim=32, T=100).to(device)

# Load your trained weights
vae.load_state_dict(torch.load("vae_tennis_serve.pth", map_location=device))
vae.eval()

def evaluate_serve(video_path, model_path="graph_opt.pb"):
    raw_seq = extract_keypoints(video_path, model_path=model_path)
    seq = preprocess_sequence(raw_seq, target_frames=100)
    seq_tensor = torch.tensor(seq.reshape(1,100,26), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        recon, _, _ = vae(seq_tensor)
    
    # Compute per-joint error
    error = torch.abs(recon - seq_tensor.view(1,100,13,2)).cpu().numpy()
    joint_error = error.sum(axis=3)  # sum x+y per joint (shape: 1,100,13)
    
    return seq, recon.cpu().numpy(), joint_error.squeeze(0)

import matplotlib.pyplot as plt

def plot_joint_error(joint_error, BODY_PARTS=BODY_PARTS):
    plt.figure(figsize=(12,5))
    for j, name in enumerate(BODY_PARTS):
        plt.plot(joint_error[:,j], label=name)
    plt.xlabel("Frame")
    plt.ylabel("Reconstruction error")
    plt.title("Joint-wise deviation from professional serve")
    plt.legend()
    plt.show()

def highlight_errors_on_video(video_path, joint_error, threshold=0.2, model_path="graph_opt.pb"):
    cap = cv.VideoCapture(video_path)
    net = cv.dnn.readNetFromTensorflow(model_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Highlight joints with error > threshold
        for j, (err) in enumerate(joint_error[frame_idx]):
            if err > threshold:
                cv.circle(frame, tuple(np.array([100+20*j,50])), 10, (0,0,255), -1)  # simple overlay

        cv.imshow("Serve Feedback", frame)
        if cv.waitKey(30) & 0xFF == 27:  # ESC to exit
            break
        frame_idx += 1
        if frame_idx >= len(joint_error):
            break
    cap.release()
    cv.destroyAllWindows()

video_path = "my_serve.mp4"
seq, recon, joint_error = evaluate_serve(video_path)

# Plot joint error over time
plot_joint_error(joint_error)

# Optional: overlay errors on video
# highlight_errors_on_video(video_path, joint_error, threshold=0.2)
