import Image
import numpy as np


def filter_image(mat, shape, order="F"):
    n = mat.shape[0]
    w = int(np.sqrt(n))
    h = int(np.ceil(1. * n / w))
    if len(shape) == 2:
        arr = np.zeros((w * (shape[0] + 1) - 1, h * (shape[1] + 1) - 1))
    elif len(shape) == 3:
        if shape[2] == 3:  # RGB image
            arr = np.zeros((w * (shape[0] + 1) - 1, h * (shape[1] + 1) - 1, 3))
        else:
            arr = np.zeros((w * (shape[0] + 1) - 1,
                            h * (shape[1] * shape[2] + 1) - 1))
    else:
        raise NotImplementedError
    i_img = 0
    for i_x in xrange(w):
        for i_y in xrange(h):
            patch = mat[i_img].reshape(shape, order=order)
            patch -= patch.min()
            patch /= patch.max()
            i1 = i_x * (shape[0] + 1)
            i2 = i1 + shape[0]
            if len(shape) == 2 or shape[2] == 3:
                j1 = i_y * (shape[1] + 1)
                j2 = j1 + shape[1]
                arr[i1:i2, j1:j2] = patch
            else:
                for i_ch in xrange(shape[2]):
                    j1 = i_y * (shape[2] * shape[1] + 1) + i_ch * shape[1]
                    j2 = j1 + shape[1]
                    arr[i1:i2, j1:j2] = patch[:, :, i_ch].squeeze()
            i_img += 1
            if i_img >= n:
                break
    image = Image.fromarray(np.uint8(255. * arr))
    return image
