"""
Download CLINC150 dataset from HuggingFace and save as JSON.

從 HuggingFace 下載 CLINC150 資料集並存成 JSON 格式。

Usage:
    python download_data.py

Output structure / 輸出結構:
    data/clinc150/
    ├── train.json          (15,250 examples)
    ├── validation.json     (3,100 examples)
    ├── test.json           (5,500 examples)
    └── intent_names.json   (151 intent names, including oos)
"""

import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not installed.")
    print("Run: pip install datasets")
    raise SystemExit(1)


OUTPUT_DIR = Path(__file__).parent / "data" / "clinc150"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading CLINC150 (plus variant) from HuggingFace...")
    print("從 HuggingFace 下載 CLINC150（plus variant）...")
    ds = load_dataset("clinc_oos", "plus")

    # Extract intent names from dataset features
    intent_names = ds["train"].features["intent"].names
    print(f"Loaded {len(intent_names)} intent names (including oos)")
    print(f"已載入 {len(intent_names)} 個 intent 名稱（含 oos）")

    with open(OUTPUT_DIR / "intent_names.json", "w", encoding="utf-8") as f:
        json.dump(intent_names, f, ensure_ascii=False, indent=2)

    # Save each split as JSON
    for split_name in ["train", "validation", "test"]:
        split_data = [
            {"text": ex["text"], "intent": ex["intent"]}
            for ex in ds[split_name]
        ]
        output_path = OUTPUT_DIR / f"{split_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}.json: {len(split_data):,} examples")

    print(f"\n✅ Done. Files saved to: {OUTPUT_DIR}")
    print(f"✅ 完成。檔案儲存於：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
