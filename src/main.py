import numpy as np
import numpy.random
import matplotlib.pyplot as plt

from .data_generation import generate_circle, generate_noise
from .ransac import Ransac

# ========== Constants ==========
NUM_CIRCLE_DATA = 100
CIRCLE_NOISE = 0.1
NUM_NOISY_DATA = 20
NOISY_NOISE = 1.
X_BOUND = 1.5
Y_BOUND = 1.5

# set random seed
np.random.seed(1234)


def generate_data():
    """
    Generates data for circle and noise

    Returns:
        Tuple[np.ndarray[float]]: the generated x, y coordinates
    """
    x_circle, y_circle = generate_circle(
        num_samples=NUM_CIRCLE_DATA,
        noise=CIRCLE_NOISE
    )

    x_noise, y_noise = generate_noise(
        num_samples=NUM_NOISY_DATA,
        noise=NOISY_NOISE,
        x_bound=X_BOUND,
        y_bound=Y_BOUND
    )

    x = np.append(x_circle, x_noise)
    y = np.append(y_circle, y_noise)
    return x, y


def run_ransac(x, y, save_file_name, is_baysac=False):
    ransac = Ransac(x, y, 50, CIRCLE_NOISE, is_baysac=is_baysac)
    ransac.execute_ransac()

    print(ransac.get_min_average_dist())

    best_model = ransac.get_best_model()
    c_x, c_y, r = best_model[0], best_model[1], best_model[2]
    x_inlier, y_inlier = zip(*list(ransac.get_inliers()))

    plt.scatter(
        x, y,
        c="blue",
        marker=".",
        label="data"
    )
    plt.scatter(
        x_inlier, y_inlier,
        facecolors='none',
        edgecolors='lime',
        label="inliers"
    )
    circle = plt.Circle(
        (c_x, c_y),
        radius=r,
        color="red",
        fill=False
    )
    plt.gca().add_patch(circle)
    plt.axis("scaled")
    plt.savefig("./out/{0}.png".format(save_file_name))
    plt.show()
    plt.clf()
    return ransac


if __name__ == "__main__":
    # data generation
    x, y = generate_data()

    # ransac
    ransac = run_ransac(x, y, "ransac", is_baysac=False)

    # baysac
    baysac = run_ransac(x, y, "baysac", is_baysac=True)
