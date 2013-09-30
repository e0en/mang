import Image
import numpy as np


def filter_image(mat, shape=None, order="F"):
    if shape is None:
        shape = mat.shape

    n = mat.shape[0]
    w = int(np.sqrt(n))
    h = int(np.ceil(1. * n / w))
    if len(shape) == 2:
        arr = np.zeros((w * (shape[0] + 1) - 1, h * (shape[1] + 1) - 1))
    elif len(shape) == 3:
        if shape[2] == 1:
            arr = np.zeros((w * (shape[0] + 1) - 1, h * (shape[1] + 1) - 1))
        elif shape[2] == 3:  # RGB image
            arr = np.zeros((w * (shape[0] + 1) - 1, h * (shape[1] + 1) - 1, 3))
        else:
            w_p = int(np.sqrt(shape[2]))
            h_p = int(np.ceil(1. * shape[2] / w_p))

            arr = np.zeros((w * ((shape[0] + 1) * w_p + 2) - 3,
                            h * ((shape[1] + 1) * h_p + 2) - 3))
    else:
        raise NotImplementedError
    i_img = 0
    for i_y in xrange(h):
        for i_x in xrange(w):
            patch = mat[i_img].reshape(shape, order=order)
            patch -= patch.min()
            patch /= patch.max()
            i1 = i_x * (shape[0] + 1)
            i2 = i1 + shape[0]
            if len(shape) == 2 or shape[2] == 3:
                j1 = i_y * (shape[1] + 1)
                j2 = j1 + shape[1]
                arr[i1:i2, j1:j2] = patch
            elif shape[2] == 1:
                j1 = i_y * (shape[1] + 1)
                j2 = j1 + shape[1]
                arr[i1:i2, j1:j2] = patch[:, :, 0].squeeze()
            else:
                i_ch = 0
                for i_y_p in xrange(h_p):
                    for i_x_p in xrange(w_p):
                        i1 = i_x * (w_p * (shape[0] + 1) + 2) + \
                            i_x_p * (shape[0] + 1)
                        i2 = i1 + shape[0]
                        j1 = i_y * (h_p * (shape[1] + 1) + 2) + \
                            i_y_p * (shape[1] + 1)
                        j2 = j1 + shape[1]
                        arr[i1:i2, j1:j2] = patch[:, :, i_ch].squeeze()
                        i_ch += 1
                        if i_ch >= shape[2]:
                            break
            i_img += 1
            if i_img >= n:
                break
    if len(arr.shape) == 2:
        arr = arr.T
    else:
        arr = np.transpose(arr, (1, 0, 2))
    image = Image.fromarray(np.uint8(255. * arr))
    return image
