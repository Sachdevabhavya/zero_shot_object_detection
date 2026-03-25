# ==========================================
#  Zero-Shot 3D Object Detection Pipeline
# ==========================================

PYTHON = python3
PIP = pip3
REQUIREMENTS = requirements.txt

# --- Variables (Override these in terminal) ---
IMAGE ?= data/input/test_image.jpg
PROMPT ?= "sofa"
OUTPUT_DIR ?= data/output
CONFIDENCE ?= 0.0001
MODEL ?= all
EPOCHS ?= 10

.PHONY: setup install download_models download_dataset run train clean help

.ONESHELL:

help:
	@echo "----------------------------------------------------------------"
	@echo "  make setup			 - Install dependencies & create folders"
	@echo "  make download_models   - Download SAM weights (Run once)"
	@echo "  make download_dataset  - Download & extract NYU Prompt 331 dataset"
	@echo "  make run			   - Run inference (generate 3D)"
	@echo "  make train			 - Train/Fine-tune models locally"
	@echo "  make clean			 - Remove output files and caches"
	@echo "----------------------------------------------------------------"

setup:
	@echo "[*] Creating Project Structure..."
	@$(PYTHON) -c "import os; [os.makedirs(d, exist_ok=True) for d in ['data/input', 'data/output', 'config', 'models/checkpoints/trained', 'data/nyu_prompt_331/images', 'data/nyu_prompt_331/depth_maps', 'data/nyu_prompt_331/masks', 'data/nyu_prompt_331/labels']]"
	@echo "[*] Installing Requirements..."
	$(PIP) install -r $(REQUIREMENTS)
	@echo "[*] Setup Complete."

download_models:
	@echo "[*] Downloading SAM weights..."
	curl -L -o models/checkpoints/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

download_dataset:
	@echo "[*] Downloading NYU Depth V2 labeled dataset (~2.8 GB)..."
	@mkdir -p data/nyu_prompt_331/images data/nyu_prompt_331/depth_maps data/nyu_prompt_331/masks data/nyu_prompt_331/seg13 data/nyu_prompt_331/seg40
	@if [ ! -s data/nyu_depth_v2_labeled.mat ]; then curl -L -C - -o data/nyu_depth_v2_labeled.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat || true; fi
	@if [ ! -s data/classMapping40.mat ]; then curl -L -o data/classMapping40.mat https://github.com/VainF/nyuv2-python-toolkit/raw/master/classMapping40.mat; fi
	@if [ ! -s data/class13Mapping.mat ]; then curl -L -o data/class13Mapping.mat https://github.com/VainF/nyuv2-python-toolkit/raw/master/class13Mapping.mat; fi
	@if [ ! -s data/splits.mat ]; then curl -L -o data/splits.mat https://github.com/VainF/nyuv2-python-toolkit/raw/master/splits.mat; fi
	@echo "[*] Extracting 331 samples (images / depth / seg13 / seg40)..."
	@$(PYTHON) scripts/extract.py
	@echo "[*] Dataset ready in data/nyu_prompt_331/"
	
run:
	@echo "[*] Running Zero-Shot 3D Pipeline..."
	@echo "	Input: $(IMAGE)"
	@echo "	Prompt: $(PROMPT)"
	$(PYTHON) main_inference.py --image_path "$(IMAGE)" --prompt "$(PROMPT)" --output_dir "$(OUTPUT_DIR)"

train:
	@echo "[*] Starting Training for model(s): $(MODEL)"
	@echo "[*] Epochs: $(EPOCHS)"
	$(PYTHON) main_train.py --model $(MODEL) --epochs $(EPOCHS)

clean:
	@$(PYTHON) -c "import shutil, pathlib; [p.unlink() for p in pathlib.Path('$(OUTPUT_DIR)').glob('*.ply')]; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	@echo "[*] Cleaned."