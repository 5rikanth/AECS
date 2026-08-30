#!/usr/bin/env bash
set -e

ENV_NAME="indicconformer"
NEMO_COMMIT="8dce88cf8e94963e2033c3137f7b9993b51db88a"

echo "======================================"
echo " AECS Environment Setup"
echo "======================================"

echo "[1/6] Checking Conda..."
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: Conda is not installed."
    exit 1
fi

eval "$(conda shell.bash hook)"

echo "[2/6] Creating Conda environment..."

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Environment already exists."
else
    conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"

echo "[3/6] Installing PyTorch CUDA 12.1..."

pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[4/6] Installing AECS dependencies..."

pip install \
    pytorch-lightning==2.1.4 \
    torchmetrics==1.2.1 \
    setuptools==80.10.2 \
    numpy==1.26.4 \
    transformers==4.40.2 \
    huggingface-hub==0.23.2 \
    datasets==5.0.1 \
    sentencepiece \
    librosa \
    lhotse \
    pyannote.core \
    IPython \
    inflect \
    pandas \
    jiwer \
    SpeechRecognition \
    soundfile \
    scipy \
    matplotlib

echo "[5/6] Getting AI4Bharat NeMo..."

if [ -d "NeMo/.git" ]; then
    echo "NeMo directory already exists."
else
    git clone https://github.com/AI4Bharat/NeMo.git
fi

cd NeMo
git fetch --all
git checkout "$NEMO_COMMIT"

echo "[6/6] Installing NeMo..."

pip install -e .

cd ..

echo
echo "======================================"
echo " Verifying installation"
echo "======================================"

python - <<'PY'
import torch
import numpy

print("Python OK")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA:", torch.version.cuda)
print("NumPy:", numpy.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python -c "import nemo.collections.asr as nemo_asr; print('NeMo ASR: OK')"

echo
echo "======================================"
echo " SETUP COMPLETE"
echo "======================================"
echo
echo "Activate with:"
echo "conda activate indicconformer"
echo
echo "You still need to provide:"
echo "1. IndicConformer .nemo model"
echo "2. MUCS datasets"
echo "3. GramVaani dataset"
