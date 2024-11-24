from PIL import Image, ImageDraw, ImageFont


class HandlerLargeImage:
    def __init__(self, origin_image_str, patch_size=680) -> None:
        Image.MAX_IMAGE_PIXELS = 500000000
        self._origin_img = Image.open(origin_image_str).convert("RGB")
        self._patch_size = patch_size
        self._patches = []
        self._patches_img = []
        self.offset_img = []

    def split_image(self):
        width, height = self._origin_img.size
        offset_y = 0
        for top in range(0, height, self._patch_size):

            offset_x = 0
            for left in range(0, width, self._patch_size):
                box = (
                    left,
                    top,
                    min(left + self._patch_size, width),
                    min(top + self._patch_size, height),
                )
                patch = self._origin_img.crop(box)

                self._patches.append((patch, box))
                self._patches_img.append(patch)
                self.offset_img.append(
                    {
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                    }
                )
                offset_x += self._patch_size
            offset_y += self._patch_size

        return self._patches_img

    def draw_on_patches(self, text="Sample Text", font_path=None, font_size=20):
        for i, (patch, box) in enumerate(self._patches):
            draw = ImageDraw.Draw(patch)
            font = (
                ImageFont.truetype(font_path, font_size)
                if font_path
                else ImageFont.load_default()
            )
            draw.text((10, 10), f"{text} - Patch {i+1}", fill="red", font=font)
        return self._patches

    def merge_patches(self):
        new_image = Image.new("RGB", self._origin_img.size)
        for patch, box in self._patches:
            new_image.paste(patch, (box[0], box[1]))
        return new_image
