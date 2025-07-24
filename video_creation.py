import numpy as np
import cv2

# === Paths ===
npz_path = "/scratch/vamshis/FINAL/TEST_5/sim00000001/ffa/concentration_data.npz"
output_video_path = "/scratch/vamshis/FINAL/TEST_5/sim00000001/ffa/concentration_video.mp4"

# === Load Data ===
data = np.load(npz_path)
frame_keys = sorted(data.files, key=lambda x: int(x))

# === Video Settings ===
height, width = 321, 321
video_size = (width, height)

# === Use 'mp4v' codec for broad compatibility
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30, video_size, isColor=True)

# === Frame Generation and Writing ===
for key in frame_keys:
    array = data[key]
    target_size = 32100

    padded = np.zeros(target_size)
    if array.size > target_size:
        padded = array[:target_size]
    else:
        padded[:array.size] = array

    image = padded.reshape(321, 100)

    # Pad to square 321x321
    square_image = np.zeros((321, 321))
    square_image[:, :100] = image

    # Normalize to 0–255
    image_uint8 = np.clip(square_image / 0.01 * 255, 0, 255).astype(np.uint8)

    # Convert grayscale to 3-channel BGR
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)

    # Write frame to video
    out.write(image_bgr)

# === Finalize Video ===
out.release()
print(f"[SUCCESS] Video saved to: {output_video_path}")

