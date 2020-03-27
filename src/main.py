import numpy as np
import numpy.random
import matplotlib.pyplot as plt

from .data_generation import generate_circle, generate_noise
from .ransac import RANSAC

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


if __name__ == "__main__":
    # data generation
    x, y = generate_data()
    plt.scatter(x, y, c="blue", marker=".", label="data")

    # ===== RANSAC =====
    ransac = RANSAC(x, y, 50, CIRCLE_NOISE, baysac=False)
    ransac.execute_ransac()

    c_x, c_y, r = ransac.best_model[0], ransac.best_model[1], ransac.best_model[2]

    circle = plt.Circle((c_x, c_y), radius=r, color="r", fc="y", fill=False)
    plt.gca().add_patch(circle)
    plt.axis("scaled")
    plt.show()
    plt.clf()

    # ===== BAYSAC =====
    # baysac = RANSAC(x, y, 50, CIRCLE_NOISE, baysac=True)

    # baysac.execute_ransac()

    # c_x, c_y, r = baysac.best_model[0], baysac.best_model[1], baysac.best_model[2]

    # circle = plt.Circle((c_x, c_y), radius=r, color="r", fc="y", fill=False)
    # plt.gca().add_patch(circle)
    # plt.axis("scaled")
    # plt.show()
    # plt.clf()
