from PIL import Image
import os 

Image.MAX_IMAGE_PIXELS = None

input_dir = r"Non_DTW_Normalization"
output_dir = r"Non_DTW_Normalization\compressed"

for file in os.listdir(input_dir):
    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # Skip non-image files

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_path = os.path.join(input_dir, file)
    print(f"Processing {input_path}")

    output_path = os.path.join(output_dir, file)
    output_path = output_path.replace('.jpg', '_resized.jpg')
    print(f"Creating {output_path}")

    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        exit(1)

    with Image.open(input_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Resize image to half the size
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)

        # Save with lower quality
        img.save(output_path, dpi=(300, 300), quality=70, optimize=True)

    print("Done!")   



