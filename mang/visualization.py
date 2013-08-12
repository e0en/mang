import Image
import ImageDraw
import numpy as np

def matrixImage(mat, patchwise=True, color=False, **kwargs):
    print 'please use matrix_image instead.'
    return matrix_image(mat, patchwise, color, **kwargs)

def matrix_image(mat, patchwise=True, color=False, **kwargs):
    min_val = kwargs['min'] if 'min' in kwargs else mat.min()
    max_val = kwargs['max'] if 'max' in kwargs else mat.max()
    mat = mat.clip(min_val, max_val)

    if 'shape' in kwargs:
        (N, dim) = mat.shape
        if patchwise:
            mat_img = (mat.T - mat.min(1)).T
            mat_img_max = mat_img.max(1) + 1.0*(mat_img.max(1) == 0)
            mat_img = (mat_img.T*(255.0/mat_img_max)).T
        else:
            mat_img = mat - mat.min()
            mat_img = mat_img*255.0/mat_img.max()
        shape = kwargs['shape']
        w = kwargs['width'] if 'width' in kwargs \
                else np.ceil(np.sqrt(N)/10)*10
        h = np.ceil(N/w)
        (w, h) = (int(w), int(h))
        if N < w:
            (w, h) = N, 1

        stride = kwargs['stride'] if 'stride' in kwargs else 0
        bgcolor = kwargs['background'] if 'background' in kwargs else 0.

        if not color:
            img = bgcolor*np.ones(((shape[0] + stride)*h + stride,
                    (shape[1] + stride)*w + stride))
            k = 0
            for y in xrange(h):
                for x in xrange(w):
                    if k >= N:
                        break
                    i1 = (shape[0] + stride)*y + stride
                    i2 = i1 + shape[0]
                    j1 = (shape[1] + stride)*x + stride
                    j2 = j1 + shape[1]

                    img[i1:i2, j1:j2] = mat_img[k,:].reshape(*shape)

                    k += 1
        else:
            img = bgcolor*np.ones(((shape[0] + stride)*h + stride,
                    (shape[1] + stride)*w + stride, 3))
            k = 0
            dim = shape[0]*shape[1]
            for y in xrange(h):
                i1 = (shape[1] + stride)*y + stride
                i2 = i1 + shape[1]
                for x in xrange(w):
                    if k >= N:
                        break
                    j1 = (shape[0] + stride)*x + stride
                    j2 = j1 + shape[0]

                    patch_data = mat_img[k, :].reshape(dim, 3)
                    if patchwise:
                        patch_data = patch_data - patch_data.min(0)
                        patch_data = patch_data/patch_data.max(0)*255
                    for ch in xrange(3):
                        img[i1:i2, j1:j2, ch] = \
                                patch_data[:, ch].reshape(shape)
                    k += 1

    else:
        mat_img = mat - mat.min()
        mat_img_max = mat_img.max()
        if mat_img_max == 0:
            mat_img_max = 1
        mat_img *= 255.0/mat_img_max
        img = mat_img
    return Image.fromarray(np.uint8(img))

def histogram(X, **kwargs):
    (N, scale) = kwargs['shape'] if 'shape' in kwargs else (100, 50)
    bins = np.zeros((N, ))
    hist_min = kwargs['min'] if 'min' in kwargs else X.min()
    hist_max = kwargs['max'] if 'max' in kwargs else X.max()
    as_image = kwargs['as_image'] if 'as_image' in kwargs else False
    bin_size = 1.0*(hist_max - hist_min)/N

    for x in X:
        i = int((x - hist_min)/bin_size)
        if i == N:
            i = N - 1
        bins[i] += 1
    bins *= scale/bins.max()

    if as_image:
        bg_color = (255, 255, 255, 0)
        fg_color = (0, 0, 0, 255)
        img = Image.new('RGBA', (N, scale), bg_color)
        draw = ImageDraw.Draw(img)
        for i in xrange(N):
            if bins[i] > 0:
                draw.line((i, scale - 1, i, scale - 1 - bins[i]),
                        fill=fg_color)
        return img
    else:
        return bins
