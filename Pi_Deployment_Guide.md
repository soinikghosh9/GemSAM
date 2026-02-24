# MedGamma Edge Deployment Guide (Raspberry Pi 5 + Hailo-10H)

Step-by-step instructions for running the MedGamma inference pipeline on the Raspberry Pi 5 with Hailo AI HAT+2 NPU acceleration.

---

## 0. System Preparation (CRITICAL — Do This First)

MedGemma is a 4-billion parameter model. Even in FP16 (~4 GB), it needs most of your Pi's 8 GB RAM. You **must** increase swap and free memory before attempting model loading.

### Increase Swap to 4 GB
```bash
# Disable any existing swap
sudo swapoff -a

# Create a 4 GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make it persist across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h
# Expected: Swap: 4.0Gi
```

### Free System Memory
```bash
# Close all browsers, desktop apps, and unnecessary services
# Kill desktop environment if possible:
sudo systemctl stop lightdm   # (optional, saves ~500 MB)
```

### Verify Readiness
```bash
free -h
# You should see:
#   Mem:  7.6Gi total, ~5.0Gi available
#   Swap: 4.0Gi
```

> **Why does it hang without swap?** The old code used `bitsandbytes` INT4 quantization, which requires **CUDA** (NVIDIA GPU). On the Pi's ARM CPU, this silently hangs. The new code uses FP16 loading (~4 GB), which works on CPU but needs sufficient RAM + swap.

---

## 1. Environment Setup

```bash
# Navigate to project
cd /home/ai-pi/Downloads/Medgamma

# Activate virtual environment
source venv/bin/activate

# Set Hugging Face cache to external drive
export HF_HOME="/media/ai-pi/One Touch1/huggingface_cache"
```

---

## 2. Running Full Inference

```bash
python edge/edge_inference.py \
    --image demo/examples/chest_xray_1.jpg \
    --task full \
    --checkpoint /home/ai-pi/Downloads/Medgamma/checkpoints/production/medgemma/detection_best \
    --hailo-hef edge/models/hailo_encoder.hef
```

### Arguments
| Argument | Description |
|---|---|
| `--image` | Path to the medical image (X-ray, MRI, CT, ultrasound) |
| `--task` | `full` (detect + segment), `detect`, `screen`, or `segment` |
| `--checkpoint` | LoRA adapter directory (use `detection_best` for best accuracy) |
| `--hailo-hef` | Hailo HEF file for NPU-accelerated SAM2 segmentation |

---

## 3. What to Expect

| Stage | Time | Description |
|---|---|---|
| Memory check | Instant | Prints RAM/swap status, warns if insufficient |
| Model loading | 5–12 min | Loads MedGemma FP16 (~4 GB) from external drive |
| Detection | 60–120s | CPU inference (token generation) |
| Unload MedGemma | ~2s | Frees RAM for segmentation |
| Segmentation | 25–50ms | Hailo NPU accelerated (or ~30s CPU fallback) |

**Total: ~7–15 minutes** (first run), ~2–3 min on subsequent runs if model stays cached.

---

## 4. Running the Quick Demo (No Model Required)

For a quick test that works **without** loading MedGemma:
```bash
python edge/quick_demo.py --image demo/examples/chest_xray_1.jpg
```
This uses simulated analysis to verify the visualization pipeline and Hailo NPU work.

---

## 5. Running the Gradio Web Demo

For an interactive web interface:
```bash
# Ensure environment is set up (Step 1)
python demo/gradio_demo.py
```
Then open `http://<Pi-IP>:7860` in a browser on any device on the same network.

> **Note:** Gradio demo loads MedGemma on first image submission, not at startup. First analysis will take ~10 min.

---

## 6. Troubleshooting

| Error | Fix |
|---|---|
| **Hangs at "Loading MedGemma"** | Swap < 4 GB. Follow Step 0 to increase swap. |
| **`ModuleNotFoundError: hailo_platform`** | Run `source venv/bin/activate` or check `include-system-site-packages = true` in `venv/pyvenv.cfg`. |
| **`OSError: PermissionError` on `/media/ai-pi/`** | External drive remounted with different name. Check `ls /media/ai-pi/` and update `HF_HOME`. |
| **`Hailo HEF not found`** | `--hailo-hef` path incorrect. Verify file exists at specified path. |
| **`RuntimeError: ... float16 ...`** | Model weights dtype mismatch. Ensure using the updated `edge_inference.py` (FP16 version). |

### Monitor Memory in Real Time
Open a second terminal:
```bash
watch -n 1 free -h
```
