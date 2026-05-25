import shutil
from pathlib import Path

INPUT_DIR = Path("data/raw/original")
OUTPUT_DIR = Path("data/selected_6_classes")

SELECTED_CLASSES = [
    "biological",
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic"
]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for nama_kelas in SELECTED_CLASSES:
        sumber = INPUT_DIR / nama_kelas
        tujuan = OUTPUT_DIR / nama_kelas

        if not sumber.exists():
            print(f"Folder tidak ditemukan: {sumber}")
            continue

        if tujuan.exists():
            shutil.rmtree(tujuan)

        shutil.copytree(sumber, tujuan)
        jumlah_gambar = len(list(tujuan.glob("*")))

        print(f"{nama_kelas}: {jumlah_gambar}")

if __name__ == "__main__":
    main()