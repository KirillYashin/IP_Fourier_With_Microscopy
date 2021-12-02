import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os


def apply_dfft(image):
    fft_image = np.fft.fft2(image)
    fft_image_shift = np.fft.fftshift(fft_image)
    return fft_image_shift


def hist_dfft(fft_image):
    mag = np.abs(fft_image)
    mag = np.log(mag+1)
    return (mag-mag.min()) * 255 / (mag.max() - mag.min())


def reverse_dfft(fft_image):
    f_inv_shift = np.fft.ifftshift(fft_image)
    reverse_image = np.real(np.fft.ifft2(f_inv_shift))
    return reverse_image


def show_dfft(img, fft, name):
    magnitude = hist_dfft(fft)

    plt.figure(figsize=(12, 6), dpi=110)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gist_gray')
    plt.title('Input:' + name)

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude, cmap='gray', vmax=255, vmin=0)
    plt.title("Magnitude Spectrum")

    plt.savefig(f'comparison/{str(name)}')
    plt.show()
    k = cv.waitKey(0)
    if k == ord('q') or k == ord('й'):
        cv.destroyAllWindows()


def compare_images(img1, img2, name, dir):
    plt.figure(figsize=(12, 6), dpi=110)
    plt.title(name)

    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap="gray", vmax=255, vmin=0)

    plt.subplot(1, 2, 2)
    plt.imshow(img2, cmap='gray', vmax=255, vmin=0)
    path = os.path.join(dir, name + '.png')
    plt.savefig(path)
    plt.show()
    k = cv.waitKey(0)
    if k == ord('q') or k == ord('й'):
        cv.destroyAllWindows()


def notch_filter(img, image_path, border_value=0.9997):
    img_fshift = apply_dfft(img)
    source = np.copy(img_fshift)

    border = np.quantile(np.abs(img_fshift), border_value)
    shift_shape = img_fshift.shape
    shape = img.shape

    center = [shape[0] // 2 - 10, shape[0] // 2 + 10,
              shape[1] // 2 - 10, shape[1] // 2 + 10]
    for x in range(0, shift_shape[0]):
        for y in range(0, shift_shape[1]):
            if np.abs(img_fshift[x][y]) > border \
                    and not (center[0] < x < center[1] and center[2] < y < center[3]):
                img_fshift[x, y] = 0

    compare_images(hist_dfft(source), hist_dfft(img_fshift), image_path, 'frequency limit')
    reverse_img = reverse_dfft(img_fshift)
    return reverse_img


if __name__ == '__main__':
    for image_path in os.listdir('stripes' + os.path.sep):
        img = np.float32(cv.imread(os.path.join('stripes', image_path), 0))
        fft_image = apply_dfft(img)
        show_dfft(img, fft_image, image_path)
        cv.imwrite(os.path.join('histogram', image_path + '.png'), hist_dfft(fft_image))

    for image_path in os.listdir('stripes' + os.path.sep):
        img = np.float32(cv.imread(os.path.join('stripes', image_path), 0))
        filtered = notch_filter(img, image_path, border_value=0.9991)
        compare_images(img, filtered, image_path, 'filtered')
