import numpy as np


# counts the number of iterations until the function diverges or
# returns the iteration threshold that we check until
def mandelbrot(c, threshold):
    z = complex(0, 0)
    for iteration in range(threshold):
        z = (z * z) + c
        if abs(z) > 2:
            return iteration
    return 0


# takes the iteration limit before declaring function as convergent and
# takes the density of the atlas
# create atlas, plot mandelbrot set, display set
def mandelbrot_set(xmin=-2.25, xmax=0.75, ymin=-1.5, ymax=1.5,
                   threshold=120, xn=1000):
    yn = int((ymax - ymin) / (xmax - xmin) * xn)
    # location and size of the atlas rectangle
    real_axis = np.linspace(xmin, xmax, xn)
    imaginary_axis = np.linspace(ymin, ymax, yn)
    # 2-D array to represent mandelbrot atlas
    atlas = np.empty((yn, xn))

    # color each point in the atlas depending on the iteration count
    for ix, cx in enumerate(real_axis):
        for iy, cy in enumerate(imaginary_axis):
            c = complex(cx, cy)
            atlas[iy, ix] = mandelbrot(c, threshold)
    return atlas


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    # time to party!!
    atlas = mandelbrot_set(-0.748768, -0.748718, 0.0650619375, 0.0650900625,
                           threshold=2048, xn=1920)
    # save mandelbrot set
    plt.imsave(f'{time.time():.0f}.jpg', atlas, cmap='gnuplot2')
