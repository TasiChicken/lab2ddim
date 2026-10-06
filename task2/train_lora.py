# The assignment does not require modifying the LoRA algorithm implementation.
# This wrapper executes the unchanged official 2026 starter training script.
from _official_loader import run_official

if __name__ == "__main__":
    run_official("train_lora.py")
