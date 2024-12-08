# %%
from PIL import Image, ImageDraw, ImageFont


def concatenate_images_with_titles(image_paths, titles, output_path):
    """
    Concatenates three images horizontally and adds a title above each image.

    :param image_paths: List of paths to the three images
    :param titles: List of titles for each image
    :param output_path: Path to save the concatenated image
    """
    # Load images
    images = [Image.open(path) for path in image_paths]

    # Set font for titles (adjust path and size as needed)
    font_size = 30
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)

    # Determine dimensions for the final image
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    title_height = font_size + 5  # Space for titles

    # Create a new image with space for titles
    result_image = Image.new(
        "RGB", (total_width, max_height + title_height), color=(255, 255, 255)
    )

    # Draw titles and paste images side by side
    draw = ImageDraw.Draw(result_image)
    x_offset = 0
    for img, title in zip(images, titles):
        draw.text(
            (x_offset + img.width // 2 - draw.textsize(title, font=font)[0] // 2, 5),
            title,
            fill="black",
            font=font,
        )
        result_image.paste(img, (x_offset, title_height))
        x_offset += img.width

    # Save the concatenated image
    result_image.save(output_path)
    print(f"Concatenated image with titles saved to {output_path}")


image_files = [
    "fig/cams/bc_cam_90.png",
    "fig/cams/ft_cam_89.png",
    "fig/cams/bl_cam_85.png",
]
titles = [
    "BC",
    "FT",
    "BL",
]
output_file = "concatenated_image.jpg"
concatenate_images_with_titles(image_files, titles, output_file)
