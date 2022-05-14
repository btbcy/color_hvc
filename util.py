import numpy as np
import matplotlib.pyplot as plt


def convert_uint8(img_in):
    _check_255(img_in)
    img_out = img_in.astype(np.uint8)
    _check_uint8(img_out)
    return img_out


def convert_bool_to_uint8(img_in):
    check_binary(img_in, bmax=1)
    img_out = img_in.astype(np.uint8) * 255
    return img_out


def convert_binary(img_in):
    check_binary(img_in, bmax=255)
    img_out = np.zeros(img_in.shape, dtype=bool)
    img_out[img_in > 0] = True
    check_binary(img_out, bmax=1)
    return img_out


def check_binary(img_in, bmax):
    assert bmax == 1 or bmax == 255
    assert len(np.unique(img_in)) <= 2
    element_max = np.max(img_in)
    assert element_max == bmax or element_max == 0


def check_img(img_in):
    _check_255(img_in)
    _check_uint8(img_in)


def _check_255(img_in):
    assert np.all(img_in < 256)
    assert np.all(img_in >= 0)


def _check_uint8(img_in):
    assert img_in.dtype == np.uint8


def show_img(img_show, title=None, figsize=(6, 6), remove_axis=True, show_immediately=False):
    plt.figure(figsize=figsize)
    if img_show.ndim == 2:
        plt.imshow(img_show, cmap='gray', vmax=255, vmin=0)
    elif img_show.ndim == 3:
        plt.imshow(img_show[:, :, ::-1], cmap='gray', vmax=255, vmin=0)
    if title:
        plt.title(title)
    if remove_axis:
        plt.axis('off')
    if show_immediately:
        plt.show()


def cal_psnr(img_origin, img_modify):
    loss = (img_origin.astype(float) - img_modify.astype(float)) ** 2
    mse = np.sum(loss) / img_origin.size
    psnr_out = 10 * np.log10(255 ** 2 / mse)
    return psnr_out
