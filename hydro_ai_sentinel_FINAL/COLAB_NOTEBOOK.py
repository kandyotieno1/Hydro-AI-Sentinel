# ============================================================
# HYDRO-AI SENTINEL — Google Colab Notebook
# Kandy Otieno Genga | SCT213-C002-0106/2022
# BSc. Data Science — JKUAT 2026
# ------------------------------------------------------------
# HOW TO USE:
#   1. Upload this file to Google Colab
#   2. Set runtime to T4 GPU (Runtime > Change runtime type > T4 GPU)
#   3. Run each cell in order (Shift+Enter)
# ============================================================

# ─── CELL 1: Install packages ───────────────────────────────
# Run this first. Click the play button or press Shift+Enter.
# Paste into a Colab code cell:
"""
!pip install tensorflow scikit-learn pandas numpy plotly -q
print("✅ Packages installed")
"""

# ─── CELL 2: Generate Data ──────────────────────────────────
# Paste the ENTIRE contents of generate_data.py into a cell, then add:
"""
main()
"""

# ─── CELL 3: Preprocess ─────────────────────────────────────
# Paste the ENTIRE contents of preprocess.py into a cell, then add:
"""
for b in ["nzoia", "tana"]:
    run(b)
"""

# ─── CELL 4: Train Model ────────────────────────────────────
# Paste the ENTIRE contents of train_model.py into a cell, then add:
"""
import os
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
results = []
for basin in ["nzoia", "tana"]:
    results.append(train_basin(basin))
for r in results:
    print(f"{r['basin'].upper()}: RMSE={r['rmse']:.4f}m  R²={r['r2']:.4f}")
"""

# ─── CELL 5: Download models folder ─────────────────────────
# After training, download everything with:
"""
import shutil
shutil.make_archive("hydro_models", "zip", ".", "models")
shutil.make_archive("hydro_outputs", "zip", ".", "outputs")
shutil.make_archive("hydro_data", "zip", ".", "data")

from google.colab import files
files.download("hydro_models.zip")
files.download("hydro_outputs.zip")
files.download("hydro_data.zip")
print("✅ Downloads started!")
"""
