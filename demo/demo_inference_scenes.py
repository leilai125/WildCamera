"""Estimate intrinsics for images in <dataset_root>/sequences/<scene>/image_2/."""

import argparse
from pathlib import Path
import sys


project_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    dataset_root = parser.parse_args().dataset_root.expanduser().resolve()

    image_folders = sorted(path for path in (dataset_root / 'sequences').glob('*/image_2') if path.is_dir())
    if not image_folders:
        parser.error(f'No sequences/<scene>/image_2/ folders found in {dataset_root}')
    output_dir = dataset_root / 'wildcam_K'
    for folder in image_folders:
        output_path = output_dir / f'{folder.parent.name}.txt'
        if output_path.exists():
            parser.error(f'Output already exists: {output_path}. Remove it before rerunning.')

    import torch
    from PIL import Image
    from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF

    # Same model setup and checkpoint loading as demo_inference.py.
    model = NEWCRFIF(version='large07', pretrained=None)
    model.eval()
    model.cuda()
    ckpt_path = project_dir / 'model_zoo/Release/wild_camera_all.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in image_folders:
        scene = folder.parent.name
        images = sorted(path for path in folder.iterdir() if path.is_file()
                        and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'})
        if not images:
            print(f'{scene}: no images, skipping', flush=True)
            continue

        output_path = output_dir / f'{scene}.txt'
        temporary_path = output_path.with_suffix('.txt.tmp')
        try:
            with temporary_path.open('w', encoding='utf-8') as stream:
                for index, images_path in enumerate(images, start=1):
                    with Image.open(images_path) as image:
                        # Repeat grayscale values across RGB channels for the network.
                        rgb = image.convert('RGB')
                        intrinsic, _ = model.inference(rgb, wtassumption=False)
                    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
                    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
                    stream.write(f'{fx:.9f}\t{fy:.9f}\t{cx:.9f}\t{cy:.9f}\n')
                    if index % 100 == 0 or index == len(images):
                        print(f'{scene}: {index}/{len(images)} images', flush=True)
            temporary_path.replace(output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f'Saved {output_path}', flush=True)


if __name__ == '__main__':
    main()
