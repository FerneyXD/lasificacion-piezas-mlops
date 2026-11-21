"""Etapa 1 (DVC): divide data/raw/<clase>/* en train/val/test estratificado."""
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from common import list_images, load_params


def main():
    p = load_params()["data"]
    raw, out = Path(p["raw_dir"]), Path(p["processed_dir"])
    if out.exists():
        shutil.rmtree(out)

    classes = sorted(d.name for d in raw.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"No se encontraron carpetas de clase en {raw}. "
                         "Estructura esperada: data/raw/<clase>/*.jpg")

    for cls in classes:
        files = list_images(raw / cls)
        if len(files) < 10:
            print(f"[aviso] clase {cls} tiene solo {len(files)} imagenes")
        train, rest = train_test_split(files, test_size=p["val_size"] + p["test_size"],
                                       random_state=p["seed"])
        val, test = train_test_split(rest, test_size=p["test_size"] / (p["val_size"] + p["test_size"]),
                                     random_state=p["seed"])
        for split, items in (("train", train), ("val", val), ("test", test)):
            dest = out / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, dest / f.name)
        print(f"{cls}: train={len(train)} val={len(val)} test={len(test)}")


if __name__ == "__main__":
    main()
