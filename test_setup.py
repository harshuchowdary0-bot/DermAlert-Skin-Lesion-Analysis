import cv2
import numpy as np

# 1. Load the raw image
image_path = "test.jpg"
img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found!")
    exit()

# --- STEP 1: CLAHE CONTRAST ENHANCEMENT ---
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y_channel, cr, cb = cv2.split(ycrcb)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl_y_channel = clahe.apply(y_channel)
merged_ycrcb = cv2.merge((cl_y_channel, cr, cb))
enhanced_img = cv2.cvtColor(merged_ycrcb, cv2.COLOR_YCrCb2BGR)

# --- STEP 2: DULL-RAZOR HAIR REMOVAL ---
gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
kernel_hair = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_hair)
_, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
shaved_img = cv2.inpaint(enhanced_img, hair_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
print("Pre-processing complete...")

# --- STEP 3: K-MEANS COLOR CLUSTERING SEGMENTATION ---
# Reshape the image pixels into a flat list of RGB values
pixel_values = shaved_img.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Define criteria and number of clusters (K=3: Skin, Spots, Shadow/Hair)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
_, labels, centers = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert centers back to 8-bit values
centers = np.uint8(centers)

# Find the cluster index that has the lowest "Green" value relative to Red (most red/inflamed)
# Centers are in BGR format: centers[:, 2] is Red, centers[:, 1] is Green
redness = centers[:, 2].astype(int) - centers[:, 1].astype(int)
spot_cluster_idx = np.argmax(redness)

# Create the binary mask for just the spot cluster
labels = labels.flatten()
segmented_image = np.zeros(labels.shape, dtype=np.uint8)
segmented_image[labels == spot_cluster_idx] = 255

# Reshape back to original image dimensions
final_mask = segmented_image.reshape(shaved_img.shape[:2])

# Quick morphological cleaning to smooth the spots
kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_clean)

print("Adaptive K-Means color segmentation complete.")

# Save your new targeted output
cv2.imwrite("final_shaved_skin.jpg", shaved_img)
cv2.imwrite("segmentation_mask.jpg", final_mask)
print("Success! Cleaned K-Means 'segmentation_mask.jpg' has been saved.")