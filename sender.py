import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import skew, kurtosis

# Define dataset paths
original_path = r"DatasetSender"
tampered_path = r"DatasetReceiver"

if not os.path.exists(original_path) or not os.path.exists(tampered_path):
    raise FileNotFoundError(f"❌ ERROR: One or both dataset folders not found: {original_path}, {tampered_path}")

# ✅ Extract RGB Features
def extract_rgb_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r, g, b = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]

    def safe_stat(func, array):
        return func(array.flatten()) if np.std(array) != 0 else 0  

    features = {
        "R_mean": np.mean(r), "G_mean": np.mean(g), "B_mean": np.mean(b),
        "R_std": np.std(r), "G_std": np.std(g), "B_std": np.std(b),
        "R_skew": safe_stat(skew, r), "G_skew": safe_stat(skew, g), "B_skew": safe_stat(skew, b),
        "R_kurtosis": safe_stat(kurtosis, r), "G_kurtosis": safe_stat(kurtosis, g), "B_kurtosis": safe_stat(kurtosis, b),
        "R_min": np.min(r), "G_min": np.min(g), "B_min": np.min(b),
        "R_max": np.max(r), "G_max": np.max(g), "B_max": np.max(b),
        "RG_Correlation": np.corrcoef(r.flatten(), g.flatten())[0, 1],
        "RB_Correlation": np.corrcoef(r.flatten(), b.flatten())[0, 1],
        "GB_Correlation": np.corrcoef(g.flatten(), b.flatten())[0, 1],
    }
    return features

# ✅ Compute SSIM Score & Generate SSIM Difference Image
def compute_ssim(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(gray1, gray2, full=True)
    
    # Normalize and convert difference to a visible image
    diff = (diff * 255).astype(np.uint8)
    diff_colormap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)  # Apply color mapping

    return score, diff_colormap

# ✅ Generate Heatmap
def generate_difference_heatmap(image1, image2):
    diff = cv2.absdiff(image1, image2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)

# ✅ Detect Tampering
def detect_tampering(features1, features2, threshold=0.02):
    return any(abs(features1[key] - features2[key]) > threshold for key in features1)

# ✅ Process Dataset
results = []
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)  # Create a folder to save SSIM images

for filename in os.listdir(original_path):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        original_image_path = os.path.join(original_path, filename)
        tampered_image_path = os.path.join(tampered_path, filename)

        if not os.path.exists(tampered_image_path):
            print(f"❌ Missing corresponding image for {filename}")
            continue

        original_image = cv2.imread(original_image_path)
        tampered_image = cv2.imread(tampered_image_path)

        if original_image is None or tampered_image is None:
            print(f"❌ Skipping invalid image: {filename}")
            continue

        # Extract features
        original_features = extract_rgb_features(original_image)
        tampered_features = extract_rgb_features(tampered_image)

        # Compute SSIM Score and SSIM Difference Image
        ssim_score, ssim_diff_image = compute_ssim(original_image, tampered_image)

        # Save SSIM Difference Image
        ssim_image_path = os.path.join(output_dir, f"SSIM_{filename}")
        cv2.imwrite(ssim_image_path, ssim_diff_image)

        # Detect tampering
        tampered = detect_tampering(original_features, tampered_features)

        # Store results
        result = {
            "Filename": filename,
            "Status": "❌ Tampered" if tampered else "✅ Not Tampered",
            "SSIM_Score": round(ssim_score, 4),
            "SSIM_Image_Path": ssim_image_path,
            **original_features,
            **tampered_features
        }
        results.append(result)

        # ✅ Display images
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("🖼️" \
        " Original Image")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(cv2.cvtColor(tampered_image, cv2.COLOR_BGR2RGB))
        plt.title("📷 Received Image")
        plt.axis("off")

        if tampered:
            heatmap = generate_difference_heatmap(original_image, tampered_image)
            plt.subplot(1, 4, 3)
            plt.imshow(heatmap)
            plt.title("🔥 Tampering Heatmap")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(cv2.cvtColor(ssim_diff_image, cv2.COLOR_BGR2RGB))
            plt.title("🔍 SSIM Difference")
            plt.axis("off")

            print(f"❌ Image {filename} is tampered! SSIM Score: {ssim_score:.4f}")
        else:
            print(f"✅ Image {filename} is NOT tampered! SSIM Score: {ssim_score:.4f}")

        plt.show()

# ✅ Save results to CSV
df = pd.DataFrame(results)
csv_path = "tampering_comparison_results.csv"
df.to_csv(csv_path, index=False)

print(f"\n✅ Results saved to {csv_path}")
print(f"✅ SSIM images saved in {output_dir}")
