# deploy.py
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin

from config.config import Config
from unet import UNet   # only needed for Option 1


# -----------------------
# 1. Model Loading
# -----------------------
def load_model(option="state_dict"):
    """Load trained model for inference."""
    if option == "state_dict":
        print("🔹 Loading model from state_dict...")
        model = UNet(in_channels=Config.IN_CHANNELS, out_channels=Config.NUM_CLASSES)
        ckpt_path = Config.SAVE_DIR / Config.BEST_MODEL_NAME
        model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
    elif option == "full":
        print("🔹 Loading full saved model...")
        ckpt_path = Config.SAVE_DIR / "best_model_full.pth"
        model = torch.load(ckpt_path, map_location=Config.DEVICE)
    else:
        raise ValueError("option must be 'state_dict' or 'full'")
    
    model.to(Config.DEVICE)
    model.eval()
    return model


# -----------------------
# 2. Preprocessing Utils
# -----------------------
def load_and_preprocess(pre_path, post_path):
    """Load pre-fire and post-fire RGB images (3 channels each)."""
    pre_fire = np.array(Image.open(pre_path).convert("RGB"), dtype=np.float32) / 255.0
    post_fire = np.array(Image.open(post_path).convert("RGB"), dtype=np.float32) / 255.0

    # Convert HWC -> CHW
    pre_fire = torch.from_numpy(pre_fire).permute(2, 0, 1)  # (3, H, W)
    post_fire = torch.from_numpy(post_fire).permute(2, 0, 1)  # (3, H, W)

    # Concatenate → (6, H, W)
    input_tensor = torch.cat([pre_fire, post_fire], dim=0).unsqueeze(0)  # add batch dim
    return input_tensor


# -----------------------
# 3. Inference
# -----------------------
def run_inference(model, input_tensor):
    """Run forward pass and return predicted mask."""
    with torch.no_grad():
        outputs = model(input_tensor.to(Config.DEVICE))
        preds = torch.argmax(outputs, dim=1).cpu().squeeze(0).numpy()
    return preds


# -----------------------
# 4. Visualization
# -----------------------
def visualize_results(pre_path, post_path, preds):
    """Show pre-fire, post-fire, and predicted mask side by side."""
    burn_cmap = plt.cm.get_cmap("jet", Config.NUM_CLASSES)
    class_names = ['Unburned', 'Low', 'Moderate', 'High', 'Cloud/No-data']

    pre_fire = Image.open(pre_path).convert("RGB")
    post_fire = Image.open(post_path).convert("RGB")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pre_fire)
    axes[0].set_title("Pre-fire")
    axes[0].axis("off")

    axes[1].imshow(post_fire)
    axes[1].set_title("Post-fire")
    axes[1].axis("off")

    im = axes[2].imshow(preds, cmap=burn_cmap, vmin=0, vmax=Config.NUM_CLASSES - 1)
    axes[2].set_title("Predicted Severity")
    axes[2].axis("off")

    cbar = plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_ticks(range(Config.NUM_CLASSES))
    cbar.set_ticklabels(class_names)

    plt.tight_layout()
    plt.show()

# -----------------------
# 5. Save as GeoTIFF
# -----------------------
def save_as_tif(preds, out_path="prediction.tif", reference_path=None, colorized=False):
    """
    Save prediction mask as GeoTIFF.
    - If colorized=False → raw integer labels (0–4).
    - If colorized=True  → RGB raster with burn_cmap applied.
    """
    height, width = preds.shape

    if colorized:
        # Burn severity colormap (RGB tuples)
        burn_colors = np.array([
            [0, 128, 0],    # green (Unburned)
            [255, 255, 0],  # yellow (Low)
            [255, 165, 0],  # orange (Moderate)
            [255, 0, 0],    # red (High)
            [128, 128, 128] # gray (Cloud/No-data)
        ], dtype=np.uint8)

        color_preds = burn_colors[preds]  # (H, W, 3)

        if reference_path:
            with rasterio.open(reference_path) as ref:
                meta = ref.meta.copy()
                meta.update({
                    "count": 3,
                    "dtype": rasterio.uint8
                })
            with rasterio.open(out_path, "w", **meta) as dst:
                for i in range(3):
                    dst.write(color_preds[:, :, i], i+1)
        else:
            transform = from_origin(0, 0, 1, 1)
            with rasterio.open(
                out_path, "w",
                driver="GTiff",
                height=height,
                width=width,
                count=3,
                dtype=rasterio.uint8,
                crs="+proj=latlong",
                transform=transform
            ) as dst:
                for i in range(3):
                    dst.write(color_preds[:, :, i], i+1)

        print(f"✅ Saved COLORIZED prediction to {out_path}")

    else:
        # Raw integer labels (0–4)
        if reference_path:
            with rasterio.open(reference_path) as ref:
                meta = ref.meta.copy()
                meta.update({
                    "count": 1,
                    "dtype": rasterio.uint8
                })
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(preds.astype(np.uint8), 1)
        else:
            transform = from_origin(0, 0, 1, 1)
            with rasterio.open(
                out_path, "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype=rasterio.uint8,
                crs="+proj=latlong",
                transform=transform
            ) as dst:
                dst.write(preds.astype(np.uint8), 1)

        print(f"✅ Saved RAW LABEL mask to {out_path}")



# -----------------------
# 6. Main (Example)
# -----------------------
if __name__ == "__main__":
    # Load model
    model = load_model(option="full")

    # Example paths (replace with your actual file paths)
    pre_path = str(Config.DATASET_PATH) + "/test/A/sample_001.png"
    post_path = str(Config.DATASET_PATH) + "/test/B/sample_001.png"

    # Prepare input
    input_tensor = load_and_preprocess(pre_path, post_path)

    # Run inference
    preds = run_inference(model, input_tensor)

    # Visualize
    visualize_results(pre_path, post_path, preds)

    # Save as GeoTIFF (uses post-fire image for metadata if available)
    save_as_tif(preds, out_path="prediction.tif", reference_path=post_path)
