from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

SOURCE_DIR = DATA_DIR / "split"

ENHANCEMENT_TYPES = [
    "clahe",
    "gaussian",
    "clahe_gaussian"
]

SPLITS = [
    "train",
    "val",
    "test"
]


def create_structure():

    classes = [
        folder.name
        for folder in (SOURCE_DIR / "train").iterdir()
        if folder.is_dir()
    ]

    enhanced_root = (
        DATA_DIR / "enhanced"
    )

    for enhancement in ENHANCEMENT_TYPES:

        for split in SPLITS:

            for class_name in classes:

                folder_path = (
                    enhanced_root
                    / enhancement
                    / split
                    / class_name
                )

                folder_path.mkdir(
                    parents=True,
                    exist_ok=True
                )

    print(
        "Struktur folder enhancement berhasil dibuat."
    )


if __name__ == "__main__":
    create_structure()

print(PROJECT_ROOT)
print(DATA_DIR)
print(SOURCE_DIR)
