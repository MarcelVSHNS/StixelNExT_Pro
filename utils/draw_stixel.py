from stixel import StixelWorld
from stixel.utils import draw_stixels_on_image
from typing import List, Optional
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description='Draw Stixels-File on given Image.')
parser.add_argument('image', help='Single image path to .png file.')
parser.add_argument('stixels', help='Single stixel .csv-file.')
parser.add_argument("-s", "--save", action="store_true", help="Saves the Stixel image.")
parser.add_argument("-nd", "--display", action="store_false", help="No Display of the Stixel image.")


def main():
    args = parser.parse_args()
    stixel_world: StixelWorld = StixelWorld.read(args.stixels)
    image: Image = Image.open(args.image)

    stixel_img = draw_stixels_on_image(image, stixel_world.stixel)
    if args.display:
        stixel_img.show()

    if args.save:
        stixel_img.save(f"Stixel_{stixel_world.image_name}")
        print(f"Image Stixel_{stixel_world.image_name} saved.")
        # stixel_world.save("")


if __name__ == "__main__":
    main()
