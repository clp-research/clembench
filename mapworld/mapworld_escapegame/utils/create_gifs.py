import os
import glob
from PIL import Image

# Frames per second for GIFs
FPS = 1.0
DURATION = int(1000 / FPS)  # duration per frame in milliseconds


def create_gif(folder_path, fps=FPS):
    """
    Reads all PNG frames in `folder_path`, stitches into a GIF named 'animation.gif'.
    """
    pattern = os.path.join(folder_path, '*.png')
    frames = sorted(glob.glob(pattern))
    if not frames:
        print(f"No frames to process in {folder_path}")
        return

    images = [Image.open(f).convert('RGB') for f in frames]
    output_path = os.path.join(folder_path, 'animation.gif')
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print(f"Created GIF at: {output_path} ({len(images)} frames, {fps} fps)")


def main(root_dir='.'):
    """
    Finds all 'combined_graphs' directories under `root_dir` and generates a GIF in each.
    """
    pattern = os.path.join(root_dir, '**', 'combined_graphs')
    for folder in glob.glob(pattern, recursive=True):
        create_gif(folder, FPS)


if __name__ == '__main__':
    main()
