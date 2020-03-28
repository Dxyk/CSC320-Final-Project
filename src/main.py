import numpy as np
import numpy.random
import matplotlib.pyplot as plt

from .data_generation import generate_circle, generate_noise
from .ransac import Ransac

# ========== Constants ==========

# set random seed
np.random.seed(1234)


def generate_data(num_circle_data=100, circle_noise=0.1,
                  num_noisy_data=20, noisy_noise=1.,
                  x_bound=1.5, y_bound=1.5):
    """
    Generates data for circle and noise

    Returns:
        Tuple[np.ndarray[float]]: the generated x, y coordinates
    """
    x_circle, y_circle = generate_circle(
        num_samples=num_circle_data,
        noise=circle_noise
    )

    x_noise, y_noise = generate_noise(
        num_samples=num_noisy_data,
        noise=noisy_noise,
        x_bound=x_bound,
        y_bound=y_bound
    )

    x = np.append(x_circle, x_noise)
    y = np.append(y_circle, y_noise)
    return x, y


def plot_ransac(x, y, fitted_ransac, show=True, save=False, save_file_name=""):
    """
    Plot the data and the fitted ransac model

    Args:
        x (np.ndarray[float]): the x coordinates of the points
        y (np.ndarray[float]): the y coordinates of the points
        fitted_ransac (Ransac): the fitted ransac / baysac model
        save_file_name (str): the basename of the image save file
    """
    best_model = fitted_ransac.get_best_model()
    c_x, c_y, r = best_model[0], best_model[1], best_model[2]
    x_inlier, y_inlier = zip(*list(fitted_ransac.get_inliers()))

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
    if save and save_file_name != "":
        plt.savefig("./out/{0}.png".format(save_file_name))
    if show:
        plt.show()
    plt.clf()
    return


def run_ransac_10_times(num_circle_data, circle_noise,
                        num_noisy_data, noisy_noise, plot=False):

    baysac_res = {"runtime": 0., "dist": 0., "inlier_dist": 0}

    ransac_runtime = []
    ransac_dist = []
    ransac_inlier_dist = []
    baysac_runtime = []
    baysac_dist = []
    baysac_inlier_dist = []

    total_num_data = num_circle_data + num_noisy_data
    for i in range(10):
        # data generation
        x, y = generate_data(num_circle_data, circle_noise,
                             num_noisy_data, noisy_noise)

        # ransac
        ransac = Ransac(x, y, 50, 0.1, is_baysac=False)
        ransac.execute_ransac()
        if plot:
            plot_ransac(x, y, ransac, show=False, save=True,
                        save_file_name="{0}_ransac".format(i))
        ransac_runtime.append(ransac.get_runtime())
        ransac_dist.append(ransac.get_total_dist() / total_num_data)
        ransac_inlier_dist.append(
            ransac.get_inlier_dist() / len(ransac.get_inliers()))

        # baysac
        baysac = Ransac(x, y, 50, 0.1, is_baysac=False)
        baysac.execute_ransac()
        if plot:
            plot_ransac(x, y, baysac, show=False, save=True,
                        save_file_name="{0}_baysac".format(i))
        baysac_runtime.append(baysac.get_runtime())
        baysac_dist.append(baysac.get_total_dist() / total_num_data)
        baysac_inlier_dist.append(
            baysac.get_inlier_dist() / len(baysac.get_inliers()))

    ransac_res = {
        "runtime": sum(ransac_runtime) / 10,
        "dist": sum(ransac_dist) / 10,
        "inlier_dist": sum(ransac_inlier_dist) / 10
    }

    baysac_res = {
        "runtime": sum(baysac_runtime) / 10,
        "dist": sum(baysac_dist) / 10,
        "inlier_dist": sum(baysac_inlier_dist) / 10
    }

    return ransac_res, baysac_res


if __name__ == "__main__":
    num_circle_data = 100
    circle_noise = 0.1
    num_noisy_data = 20
    noisy_noise = 1.
    x_bound = 1.5
    y_bound = 1.5

    ransac_runtime = []
    ransac_dist = []
    ransac_inlier_dist = []
    baysac_runtime = []
    baysac_dist = []
    baysac_inlier_dist = []

    for total_num_data in [5000]:
        print "{0} num_data: {1} {0}".format("=" * 10, total_num_data)
        for ratio in np.arange(0, 1, 0.1):
            print "{0} ratio: {1} {0}".format("=" * 5, ratio)
            num_circle_data = int(total_num_data * ratio)
            num_noisy_data = int(total_num_data * (1 - ratio))
            ransac_res, baysac_res = run_ransac_10_times(num_circle_data, circle_noise,
                                                         num_noisy_data, noisy_noise)

            print "RANSAC"
            ransac_runtime.append(ransac_res["runtime"])
            print "avg time elapsed: {0}".format(ransac_res["runtime"])
            ransac_dist.append(ransac_res["dist"])
            print "avg dist: {0}".format(ransac_res["dist"])
            ransac_inlier_dist.append(ransac_res["inlier_dist"])
            print "avg inlier dist: {0}".format(ransac_res["inlier_dist"])

            print "BAYSAC"
            baysac_runtime.append(baysac_res["runtime"])
            print "avg time elapsed: {0}".format(baysac_res["runtime"])
            baysac_dist.append(baysac_res["dist"])
            print "avg dist: {0}".format(baysac_res["dist"])
            baysac_inlier_dist.append(baysac_res["inlier_dist"])
            print "avg inlier dist: {0}".format(baysac_res["inlier_dist"])

        plt.clf()
        plt.plot(ransac_runtime, color="blue", linestyle="-",
                 marker=".", label="ransac runtime")
        plt.plot(baysac_runtime, color="red", linestyle="-",
                 marker=".", label="baysac runtime")
        plt.title("Runtime Comparison")
        plt.xlabel("Circle / Noise Ratio")
        plt.ylabel("Runtime")
        plt.legend(loc="best")
        plt.show()

        plt.clf()
        plt.plot(ransac_dist, color="blue", linestyle="-",
                 marker=".", label="ransac dist")
        plt.plot(baysac_dist, color="red", linestyle="-",
                 marker=".", label="baysac dist")
        plt.title("Dist Comparison")
        plt.xlabel("Circle / Noise Ratio")
        plt.ylabel("Dist")
        plt.legend(loc="best")
        plt.show()

        plt.clf()
        plt.plot(ransac_inlier_dist, color="blue", linestyle="-",
                 marker=".", label="ransac inlier dist")
        plt.plot(baysac_inlier_dist, color="red", linestyle="-",
                 marker=".", label="baysac inlier dist")
        plt.title("Inlier Dist Comparison")
        plt.xlabel("Circle / Noise Ratio")
        plt.ylabel("Inlier Dist")
        plt.legend(loc="best")
        plt.show()
