from rembg import remove
from PIL import Image
import os
from pathlib import Path

"""
    you should download these libraries
    
    pip install pillow
    pip install rembg
    pip install onnxruntime
"""

def image_list():
    try:
        base = Path(__file__).parent
    except:
        base = Path.cwd().parent

    mock_data_path = base / "frames"

    # SORT the directory listing
    folders_list = sorted(mock_data_path.iterdir(), key=lambda p: p.name)
    return folders_list

input_path = "input_images"      # folder or single image
output_path = "output_images"    # folder to save results
os.makedirs(output_path, exist_ok=True)

def remove_bg_from_file(file_path, output_folder):
    try:
        img = Image.open(file_path)
        result = remove(img)  # remove background
        # Save output as PNG (transparent background)
        base_name = os.path.basename(file_path)
        name, ext = os.path.splitext(base_name)
        result.save(os.path.join(output_folder, f"{name}_no_bg.png"))
        print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

# Check if input_path is a folder or single file
if os.path.isdir(input_path):
    for file_name in os.listdir(input_path):
        file_path = os.path.join(input_path, file_name)
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            remove_bg_from_file(file_path, output_path)
else:
    remove_bg_from_file(input_path, output_path)

print("All done!")
