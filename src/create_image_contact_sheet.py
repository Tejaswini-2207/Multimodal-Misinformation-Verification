
from pathlib import Path
from PIL import Image, ImageDraw
import math

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "data" / "images"
OUTPUT_DIR = BASE_DIR / "outputs" / "image_contact_sheets"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

files = sorted(
    p for p in IMAGE_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in SUPPORTED
)

if not files:
    raise FileNotFoundError("No images found in data/images.")

PAGE_SIZE = 25
CELL_W = 220
CELL_H = 200
THUMB_W = 200
THUMB_H = 155
COLUMNS = 5

for page_number in range(math.ceil(len(files) / PAGE_SIZE)):

    page_files = files[
        page_number * PAGE_SIZE:
        (page_number + 1) * PAGE_SIZE
    ]

    rows = math.ceil(len(page_files) / COLUMNS)

    sheet = Image.new(
        "RGB",
        (COLUMNS * CELL_W, rows * CELL_H),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for i, path in enumerate(page_files):

        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((THUMB_W, THUMB_H))

            col = i % COLUMNS
            row = i // COLUMNS

            x = col * CELL_W
            y = row * CELL_H

            image_x = x + (CELL_W - image.width) // 2
            image_y = y + 5

            sheet.paste(image, (image_x, image_y))

            draw.text(
                (x + 5, y + 165),
                path.name[:32],
                fill="black"
            )

        except Exception as error:
            print("Could not open:", path.name, error)

    output_path = (
        OUTPUT_DIR /
        f"contact_sheet_{page_number + 1}.jpg"
    )

    sheet.save(output_path, quality=90)

    print("Saved:", output_path)

print("\nTotal images:", len(files))
print("Contact sheets created successfully!")
