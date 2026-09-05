from ..backend import get_array_module

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C NDArray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        xp = get_array_module(img)
        flip_img = xp.random.rand() < self.p
        return xp.where(flip_img, img[:, ::-1, :], img)


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        xp = get_array_module(img)
        shifts = xp.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        H, W, C = img.shape

        padded_img = xp.pad(img,
                            ((self.padding, self.padding),
                            (self.padding, self.padding),
                            (0, 0)),
                            mode="constant",
                            constant_values=0)
        rows = xp.arange(H) + self.padding + shifts[0]
        columns = xp.arange(W) + self.padding + shifts[1]
        cropped_img = padded_img[rows[:, None], columns[None, :], :]

        return cropped_img
