import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split


INPUT_DIR = Path("data/selected_6_classes")
OUTPUT_DIR = Path("data/split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def ambil_data_gambar():
    data_gambar = []
    label_gambar = []

    for folder_kelas in INPUT_DIR.iterdir():
        if folder_kelas.is_dir():
            nama_kelas = folder_kelas.name

            for file_gambar in folder_kelas.iterdir():
                if file_gambar.suffix.lower() in IMAGE_EXTENSIONS:
                    data_gambar.append(file_gambar)
                    label_gambar.append(nama_kelas)

    return data_gambar, label_gambar


def reset_folder_output():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)


def salin_gambar(data, label, nama_split):
    for path_gambar, nama_kelas in zip(data, label):
        folder_tujuan = OUTPUT_DIR / nama_split / nama_kelas
        folder_tujuan.mkdir(parents=True, exist_ok=True)

        path_tujuan = folder_tujuan / path_gambar.name
        shutil.copy2(path_gambar, path_tujuan)


def tampilkan_jumlah_data():
    print("\nJumlah data setelah split:")

    for split in ["train", "val", "test"]:
        print(f"\n{split.upper()}")

        total = 0
        folder_split = OUTPUT_DIR / split

        for folder_kelas in sorted(folder_split.iterdir()):
            if folder_kelas.is_dir():
                jumlah = len([
                    file for file in folder_kelas.iterdir()
                    if file.suffix.lower() in IMAGE_EXTENSIONS
                ])

                total += jumlah
                print(f"{folder_kelas.name}: {jumlah}")

        print(f"Total {split}: {total}")


def main():
    data_gambar, label_gambar = ambil_data_gambar()

    if len(data_gambar) == 0:
        print("Tidak ada gambar ditemukan")
        return

    reset_folder_output()

    # Split pertama:
    # 70% train, 30% val + test
    x_train, x_temp, y_train, y_temp = train_test_split(
        data_gambar,
        label_gambar,
        test_size=0.30,
        stratify=label_gambar,
        random_state=42
    )

    # Split kedua:
    # 30% jadi 15% val dan 15% test
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    salin_gambar(x_train, y_train, "train")
    salin_gambar(x_val, y_val, "val")
    salin_gambar(x_test, y_test, "test")

    print("Split dataset 70/15/15")
    tampilkan_jumlah_data()


if __name__ == "__main__":
    main()