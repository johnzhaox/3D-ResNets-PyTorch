import random
import torch.functional as F
from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from PIL import Image


class Compose(transforms.Compose):

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(transforms.ToTensor):

    def randomize_parameters(self):
        pass


class Normalize(transforms.Normalize):

    def randomize_parameters(self):
        pass


class ScaleValue(object):

    def __init__(self, s):
        self.s = s

    def __call__(self, tensor):
        tensor *= self.s
        return tensor

    def randomize_parameters(self):
        pass


class Resize(transforms.Resize):

    def randomize_parameters(self):
        pass


class Scale(transforms.Scale):

    def randomize_parameters(self):
        pass


class CenterCrop(transforms.CenterCrop):

    def randomize_parameters(self):
        pass


class CornerCrop(object):

    def __init__(self,
                 size,
                 crop_position=None,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.size = size
        self.crop_position = crop_position
        self.crop_positions = crop_positions

        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.randomize_parameters()

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        h, w = (self.size, self.size)
        if self.crop_position == 'c':
            i = int(round((image_height - h) / 2.))
            j = int(round((image_width - w) / 2.))
        elif self.crop_position == 'tl':
            i = 0
            j = 0
        elif self.crop_position == 'tr':
            i = 0
            j = image_width - self.size
        elif self.crop_position == 'bl':
            i = image_height - self.size
            j = 0
        elif self.crop_position == 'br':
            i = image_height - self.size
            j = image_width - self.size

        img = F.crop(img, i, j, h, w)

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, crop_position={1}, randomize={2})'.format(
            self.size, self.crop_position, self.randomize)


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super().__init__(p)
        self.randomize_parameters()

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.random_p < self.p:
            return F.hflip(img)
        return img

    def randomize_parameters(self):
        self.random_p = random.random()


class MultiScaleCornerCrop(object):

    def __init__(self,
                 size,
                 scales,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br'],
                 interpolation=Image.BILINEAR):
        self.size = size
        self.scales = scales
        self.interpolation = interpolation
        self.crop_positions = crop_positions

        self.randomize_parameters()

    def __call__(self, img):
        short_side = min(img.size[0], img.size[1])
        crop_size = int(short_side * self.scale)
        self.corner_crop.size = crop_size

        img = self.corner_crop(img)
        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        crop_position = self.crop_positions[random.randint(
            0,
            len(self.crop_positions) - 1)]

        self.corner_crop = CornerCrop(None, crop_position)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, scales={1}, interpolation={2})'.format(
            self.size, self.scales, self.interpolation)


class RandomResizedCrop(transforms.RandomResizedCrop):

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=Image.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)
        self.randomize_parameters()

    def __call__(self, img):
        if self.randomize:
            self.random_crop = self.get_params(img, self.scale, self.ratio)
            self.randomize = False

        i, j, h, w = self.random_crop
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def randomize_parameters(self):
        self.randomize = True


class ColorJitter(transforms.ColorJitter):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        super().__init__(brightness, contrast, saturation, hue)
        self.p = p
        self.randomize_parameters()

    def __call__(self, img):
        if self.random_p < self.p:
            for fn_id in self.fn_idx:
                if fn_id == 0 and self.brightness_factor is not None:
                    img = F.adjust_brightness(img, self.brightness_factor)
                elif fn_id == 1 and self.contrast_factor is not None:
                    img = F.adjust_contrast(img, self.contrast_factor)
                elif fn_id == 2 and self.saturation_factor is not None:
                    img = F.adjust_saturation(img, self.saturation_factor)
                elif fn_id == 3 and self.hue_factor is not None:
                    img = F.adjust_hue(img, self.hue_factor)

        return img

    def randomize_parameters(self):
        self.random_p = random.random()
        self.fn_idx, self.brightness_factor, self.contrast_factor, \
            self.saturation_factor, self.hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation,
                            self.hue)


class PickFirstChannels(object):

    def __init__(self, n):
        self.n = n

    def __call__(self, tensor):
        return tensor[:self.n, :, :]

    def randomize_parameters(self):
        pass


class RandomShuffleChannels(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        img: PIL Image
        """
        if self.random_p < self.p:
            chls = list(img.split())
            random.shuffle(chls)
            return Image.merge("RGB", chls)
        return img

    def randomize_parameters(self):
        self.random_p = random.random()
