import numpy as np


# counts the number of iterations until the function diverges or
# returns the iteration threshold that we check until
def mandelbrot(c, threshold):
    z = complex(0, 0)
    for iteration in range(threshold):
        z = (z * z) + c
        if abs(z) > 4:
            break
    return iteration


# takes the iteration limit before declaring function as convergent and
# takes the density of the atlas
# create atlas, plot mandelbrot set, display set
def mandelbrot_set(xmin=-2.25, xmax=0.75, ymin=-1.5, ymax=1.5,
                   threshold=120, density=1000):
    # location and size of the atlas rectangle
    real_axis = np.linspace(xmin, xmax, density)
    imaginary_axis = np.linspace(ymin, ymax, density)
    # 2-D array to represent mandelbrot atlas
    atlas = np.empty((density, density))

    # color each point in the atlas depending on the iteration count
    for ix, cx in enumerate(real_axis):
        for iy, cy in enumerate(imaginary_axis):
            c = complex(cx, cy)
            atlas[iy, ix] = mandelbrot(c, threshold)
    return atlas


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # time to party!!
    atlas = mandelbrot_set(-0.22, -0.219, -0.70, -0.699)
    # plot and display mandelbrot set
    plt.imshow(atlas, interpolation="nearest")
    plt.show()
