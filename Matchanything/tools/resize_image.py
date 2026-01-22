import argparse
import os
from PIL import Image

# Disable DecompressionBombError for large images
Image.MAX_IMAGE_PIXELS = None

def resize_image(input_path, edge_ratio):
    # Load the image
    with Image.open(input_path) as img:
        # Preserve format and mode
        original_format = img.format
        original_mode = img.mode

        # Get original size
        width, height = img.size

        # Determine new size based on the longer edge
        if width >= height:
            new_width = int(width * edge_ratio)
            new_height = int((new_width / width) * height)
        else:
            new_height = int(height * edge_ratio)
            new_width = int((new_height / height) * width)

        # Resize with high-quality resampling
        resized = img.resize((new_width, new_height), Image.LANCZOS)

        # Preserve original bit depth using 'mode'
        resized = resized.convert(original_mode)

        # Build output filename
        name, ext = os.path.splitext(os.path.basename(input_path))
        output_format = original_format if original_format else ext.replace('.', '').upper()
        output_path = f"{name}_.{output_format.lower()}"

        # Save with same format and quality (if applicable)
        save_params = {}
        if output_format.upper() == "JPEG":
            save_params["quality"] = 95

        resized.save(output_path, format=output_format, **save_params)

        print(f"✅ Resized image saved to: {output_path}")
        print(f"   Original size: {width}x{height}")
        print(f"   New size:      {new_width}x{new_height}")
        print(f"   Format:        {output_format}, Mode: {original_mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize a large image based on edge ratio.")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("edge_ratio", type=float, help="Ratio to scale the longer edge (e.g. 0.5 for 50%)")

    args = parser.parse_args()
    resize_image(args.image_path, args.edge_ratio)
