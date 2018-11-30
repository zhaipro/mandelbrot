import matplotlib.pyplot as plt
import numpy as np


# counts the number of iterations until the function diverges or
# returns the iteration threshold that we check until
def count_iterations_until_divergent(c, threshold):
    z = complex(0, 0)
    for iteration in range(threshold):
        z = (z * z) + c
        if abs(z) > 4:
            break
    return iteration


# takes the iteration limit before declaring function as convergent and
# takes the density of the atlas
# create atlas, plot mandelbrot set, display set
def mandelbrot(threshold, density):
    # location and size of the atlas rectangle
    # real_axis = np.linspace(-2.25, 0.75, density)
    # imaginary_axis = np.linspace(-1.5, 1.5, density)
    real_axis = np.linspace(-0.22, -0.219, density)
    imaginary_axis = np.linspace(-0.70, -0.699, density)
    # 2-D array to represent mandelbrot atlas
    atlas = np.empty((density, density))

    # color each point in the atlas depending on the iteration count
    for ix, cx in enumerate(real_axis):
        for iy, cy in enumerate(imaginary_axis):
            c = complex(cx, cy)
            atlas[iy, ix] = count_iterations_until_divergent(c, threshold)

    # plot and display mandelbrot set
    plt.imshow(atlas, interpolation="nearest")
    plt.show()


# time to party!!
mandelbrot(120, 1000)
