try:
    import fastai
    print("✅ FastAI version:", fastai.__version__)
except Exception as e:
    print("❌ FastAI failed:", e)

try:
    from fastai.vision.all import *
    print("✅ fastai.vision.all imported successfully.")
except Exception as e:
    print("❌ fastai.vision.all failed:", e)

try:
    import pandas as pd
    print("✅ Pandas version:", pd.__version__)
except Exception as e:
    print("❌ Pandas failed:", e)

try:
    import os
    print("✅ OS module loaded. Current directory:", os.getcwd())
except Exception as e:
    print("❌ OS failed:", e)

try:
    from pathlib import Path
    print("✅ Pathlib loaded. Current path:", Path('.').resolve())
except Exception as e:
    print("❌ Pathlib failed:", e)

try:
    import torch
    print("✅ Torch version:", torch.__version__)
    print("✅ CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("✅ GPU detected:", torch.cuda.get_device_name(0))
except Exception as e:
    print("❌ Torch failed:", e)

try:
    import numpy as np
    print("✅ NumPy test:", np.array([1, 2, 3]) * 2)
except Exception as e:
    print("❌ NumPy failed:", e)

try:
    import glob
    print("✅ Glob module loaded. Python files:", glob.glob("*.py"))
except Exception as e:
    print("❌ Glob failed:", e)

try:
    import time
    print("✅ Time module loaded. Current time:", time.ctime())
except Exception as e:
    print("❌ Time failed:", e)

try:
    from datetime import timedelta
    print("✅ Timedelta loaded. Example:", timedelta(days=1))
except Exception as e:
    print("❌ Timedelta failed:", e)

try:
    import sys
    print("✅ Sys module loaded. Python version:", sys.version)
except Exception as e:
    print("❌ Sys failed:", e)

try:
    import tarfile
    print("✅ Tarfile module loaded.")
except Exception as e:
    print("❌ Tarfile failed:", e)

try:
    import tempfile
    print("✅ Tempfile module loaded. Temp dir:", tempfile.gettempdir())
except Exception as e:
    print("❌ Tempfile failed:", e)

try:
    from PIL import Image
    print("✅ PIL.Image loaded.")
except Exception as e:
    print("❌ PIL.Image failed:", e)