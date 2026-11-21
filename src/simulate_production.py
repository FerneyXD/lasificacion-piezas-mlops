"""Genera un lote de imagenes de "produccion" degradadas a partir de imagenes de test.

Ejemplos:
  python src/simulate_production.py --out data/production/batch_01 --mode none          # lote limpio
  python src/simulate_production.py --out data/production/batch_02 --mode blur --severity 2
  python src/simulate_production.py --out data/production/batch_03 --mode mixed --severity 3
  python src/simulate_production.py --out data/production/batch_02 --upload s3://<bucket>/entrada/

Modos: none | blur (desenfoque) | dark (poca luz) | noise (ruido) | rotate | mixed (combina todos)
Severidad 1-3. Escribe manifest.csv con lo aplicado a cada imagen.
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from common import list_images

BLUR = {1: 1.0, 2: 2.0, 3: 4.0}
DARK = {1: 0.7, 2: 0.5, 3: 0.3}
NOISE = {1: 8, 2: 20, 3: 40}
ROTATE = {1: 10, 2: 25, 3: 45}


def apply(img: Image.Image, mode: str, sev: int, rng: random.Random):
    ops = ["blur", "dark", "noise", "rotate"] if mode == "mixed" else ([] if mode == "none" else [mode])
    applied = []
    for op in ops:
        if op == "blur":
            img = img.filter(ImageFilter.GaussianBlur(BLUR[sev]))
        elif op == "dark":
            img = ImageEnhance.Brightness(img).enhance(DARK[sev])
        elif op == "noise":
            arr = np.asarray(img, dtype="float32")
            arr += np.random.default_rng(rng.randint(0, 10**6)).normal(0, NOISE[sev], arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
        elif op == "rotate":
            img = img.rotate(rng.uniform(-ROTATE[sev], ROTATE[sev]), resample=Image.BILINEAR)
        applied.append(f"{op}:{sev}")
    return img, applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/processed/test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--mode", default="mixed", choices=["none", "blur", "dark", "noise", "rotate", "mixed"])
    ap.add_argument("--severity", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--upload", help="s3://bucket/entrada/ (opcional, sube el lote para probar la Lambda)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = list_images(args.source)
    if not files:
        raise SystemExit(f"Sin imagenes en {args.source}")
    chosen = rng.sample(files, min(args.n, len(files)))

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    with open(out / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "original_class", "transforms"])
        for i, f in enumerate(chosen):
            img, applied = apply(Image.open(f).convert("RGB"), args.mode, args.severity, rng)
            name = f"prod_{i:04d}.jpg"  # sin la clase en el nombre, como llegaria en produccion
            img.save(out / name, quality=92)
            w.writerow([name, f.parent.name, "+".join(applied) or "none"])
    print(f"{len(chosen)} imagenes en {out} (modo={args.mode}, severidad={args.severity})")

    if args.upload:
        import boto3

        bucket, _, prefix = args.upload.removeprefix("s3://").partition("/")
        s3 = boto3.client("s3")
        for p in out.glob("*.jpg"):
            s3.upload_file(str(p), bucket, f"{prefix}{p.name}")
        print(f"Subidas a {args.upload}")


if __name__ == "__main__":
    main()
