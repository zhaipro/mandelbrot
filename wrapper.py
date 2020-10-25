# encoding : utf-8
import os
import sys

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule


class Mandelbrot:

    def __init__(self):
        with open('mandelbrot.cu', 'r') as f:
            source = f.read()
        mod = SourceModule(source)
        self._mandelbrot_set = mod.get_function('mandelbrot_set')

    def _process(self, xmin, xmax, ymin, ymax, threshold, atlas):
        yn, xn = atlas.shape
        self._mandelbrot_set(np.float64(xmin), np.float64(xmax),
                             np.float64(ymin), np.float64(ymax),
                             np.intc(xn), np.intc(yn),
                             np.intc(threshold), atlas,
                             block=(16, 16, 1), grid=(16, 16, 1))
        return atlas

    def process(self, xmin, xmax, ymin, ymax, xn, yn, threshold):
        atlas = gpuarray.zeros((yn, xn), dtype=np.int32)
        self._process(xmin, xmax, ymin, ymax, threshold, atlas)
        return atlas.get()     # ?????


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    mandelbrot = Mandelbrot()
    atlas = mandelbrot.process(-0.748768, -0.748718, 0.0650619375, 0.0650900625, xn=1920, yn=1080, threshold=2048)
    plt.imsave(f'a.png', atlas, cmap='gnuplot2')
