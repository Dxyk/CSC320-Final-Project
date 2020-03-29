import numpy as np
import numpy.random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from data_generation import generate_circle, generate_noise
from ransac import Ransac

# ========== Constants ==========

# set random seed
np.random.seed(1234)


def generate_data(num_circle_data=100, circle_noise=0.1,
                  num_noisy_data=20, noisy_noise=1.):
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
        noise=noisy_noise
    )

    x = np.append(x_circle, x_noise)
    y = np.append(y_circle, y_noise)
    return x, y


def plot_ransac(x, y, fitted_ransac, save_path=""):
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

    plt.clf()
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
    plt.margins(0, 0)
    if SAVE and save_path != "":
        plt.savefig(save_path, bbox_inches='tight',
                    pad_inches=0)
    if SHOW:
        plt.show()
    return


def plot_result(res, xs, xlabel, curr_dir):
    # runtime plot
    plt.clf()
    plt.plot(xs, res["ransac"]["runtime"], color="blue", linestyle="-",
             marker=".", label="ransac runtime")
    plt.plot(xs, res["baysac"]["runtime"], color="red", linestyle="-",
             marker=".", label="baysac runtime")
    plt.title("Runtime Comparison")
    plt.xlabel(xlabel)
    plt.ylabel("Runtime")
    plt.legend(loc="best")
    plt.margins(0, 0)
    if SAVE:
        plt.savefig(curr_dir + "runtime.png", bbox_inches='tight',
                    pad_inches=0)
    if SHOW:
        plt.show()

    # dist plot
    plt.clf()
    plt.plot(xs, res["ransac"]["dist"], color="blue", linestyle="-",
             marker=".", label="ransac dist")
    plt.plot(xs, res["baysac"]["dist"], color="red", linestyle="-",
             marker=".", label="baysac dist")
    plt.title("Dist Comparison")
    plt.xlabel(xlabel)
    plt.ylabel("Dist")
    plt.legend(loc="best")
    plt.margins(0, 0)
    if SAVE:
        plt.savefig(curr_dir + "dist.png", bbox_inches='tight',
                    pad_inches=0)
    if SHOW:
        plt.show()

    # inlier dist plot
    plt.clf()
    plt.plot(xs, res["ransac"]["inlier_dist"], color="blue", linestyle="-",
             marker=".", label="ransac inlier dist")
    plt.plot(xs, res["baysac"]["inlier_dist"], color="red", linestyle="-",
             marker=".", label="baysac inlier dist")
    plt.title("Inlier Dist Comparison")
    plt.xlabel(xlabel)
    plt.ylabel("Inlier Dist")
    plt.legend(loc="best")
    plt.margins(0, 0)
    if SAVE:
        plt.savefig(curr_dir + "inlier_dist.png", bbox_inches='tight',
                    pad_inches=0)
    if SHOW:
        plt.show()

    # accuracy plot
    plt.clf()
    plt.plot(xs, res["ransac"]["accuracy"], color="blue", linestyle="-",
             marker=".", label="ransac accuracy")
    plt.plot(xs, res["baysac"]["accuracy"], color="red", linestyle="-",
             marker=".", label="baysac accuracy")
    plt.title("Accuracy Comparison")
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.margins(0, 0)
    if SAVE:
        plt.savefig(curr_dir + "accuracy.png", bbox_inches='tight',
                    pad_inches=0)
    if SHOW:
        plt.show()
    return


def run_ransac_10_times(num_circle_data, circle_noise,
                        num_noisy_data, noisy_noise, res, num_iters=50,
                        save_dir="./out/"):
    ransac_runtime = []
    ransac_dist = []
    ransac_inlier_dist = []
    ransac_accuracy = []
    baysac_runtime = []
    baysac_dist = []
    baysac_inlier_dist = []
    baysac_accuracy = []

    total_num_data = num_circle_data + num_noisy_data
    for i in range(10):
        # data generation
        x, y = generate_data(num_circle_data, circle_noise,
                             num_noisy_data, noisy_noise)

        # ransac
        ransac = Ransac(x, y, num_iters, circle_noise, is_baysac=False)
        ransac.execute_ransac()
        if PLOT:
            plot_ransac(x, y, ransac,
                        save_path="{0}/{1}_ransac.png".format(save_dir, i))
        ransac_runtime.append(ransac.get_runtime())
        ransac_dist.append(ransac.get_total_dist() / total_num_data)
        ransac_inlier_dist.append(
            ransac.get_inlier_dist() / len(ransac.get_inliers()))
        if num_circle_data != 0:
            circle_points = set(zip(x[:num_circle_data],
                                    y[:num_circle_data]))
            ransac_num_correct = len(
                circle_points.intersection(ransac.get_inliers()))
            ransac_accuracy.append(ransac_num_correct / float(num_circle_data))
        else:
            ransac_accuracy.append(0.)

        # baysac
        baysac = Ransac(x, y, num_iters, circle_noise, is_baysac=True)
        baysac.execute_ransac()
        if PLOT:
            plot_ransac(x, y, baysac,
                        save_path="{0}/{1}_baysac.png".format(save_dir, i))
        baysac_runtime.append(baysac.get_runtime())
        baysac_dist.append(baysac.get_total_dist() / total_num_data)
        baysac_inlier_dist.append(
            baysac.get_inlier_dist() / len(baysac.get_inliers()))
        if num_circle_data != 0:
            circle_points = set(zip(x[:num_circle_data],
                                    y[:num_circle_data]))
            baysac_num_correct = len(
                circle_points.intersection(baysac.get_inliers()))
            baysac_accuracy.append(baysac_num_correct / float(num_circle_data))
        else:
            baysac_accuracy.append(0.)

    ransac_runtime = sum(ransac_runtime) / 10
    ransac_dist = sum(ransac_dist) / 10
    ransac_inlier_dist = sum(ransac_inlier_dist) / 10
    ransac_accuracy = sum(ransac_accuracy) / 10

    baysac_runtime = sum(baysac_runtime) / 10
    baysac_dist = sum(baysac_dist) / 10
    baysac_inlier_dist = sum(baysac_inlier_dist) / 10
    baysac_accuracy = sum(baysac_accuracy) / 10

    print "RANSAC"
    res["ransac"]["runtime"].append(ransac_runtime)
    print "avg time elapsed: {0}".format(ransac_runtime)
    res["ransac"]["dist"].append(ransac_dist)
    print "avg dist: {0}".format(ransac_dist)
    res["ransac"]["inlier_dist"].append(ransac_inlier_dist)
    print "avg inlier dist: {0}".format(ransac_inlier_dist)
    res["ransac"]["accuracy"].append(ransac_accuracy)
    print "avg accuracy: {0}".format(ransac_accuracy)

    print "BAYSAC"
    res["baysac"]["runtime"].append(baysac_runtime)
    print "avg time elapsed: {0}".format(baysac_runtime)
    res["baysac"]["dist"].append(baysac_dist)
    print "avg dist: {0}".format(baysac_dist)
    res["baysac"]["inlier_dist"].append(baysac_inlier_dist)
    print "avg inlier dist: {0}".format(baysac_inlier_dist)
    res["baysac"]["accuracy"].append(baysac_accuracy)
    print "avg accuracy: {0}".format(baysac_accuracy)

    return


if __name__ == "__main__":
    import os
    import copy

    # plot flags
    PLOT = True
    SHOW = False
    SAVE = True

    # run flags
    RUN_BASELINE = True
    RUN_RATIO = True
    RUN_TOTAL_NUM = True
    RUN_CIRCLE_NOISE = True
    RUN_NOISY_NOISE = True
    RUN_NUM_ITERS = True

    # empty result dict
    empty_res = {
        "ransac": {
            "runtime": [],
            "dist": [],
            "inlier_dist": [],
            "accuracy": []
        },
        "baysac": {
            "runtime": [],
            "dist": [],
            "inlier_dist": [],
            "accuracy": []
        }
    }

    # default param dict
    default_param = {
        "total_num": 1000,
        "ratio": 0.8,
        "circle_noise": 0.1,
        "noisy_noise": 1.
    }

    # baseline group
    if RUN_BASELINE:
        curr_dir = "./out/baseline/"
        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)

        res = copy.deepcopy(empty_res)

        num_circle_data = int(
            default_param["total_num"] * default_param["ratio"])
        num_noisy_data = int(
            default_param["total_num"] * (1 - default_param["ratio"]))
        run_ransac_10_times(num_circle_data, default_param["circle_noise"],
                            num_noisy_data, default_param["noisy_noise"], res,
                            save_dir=curr_dir)

    # ratio group
    if RUN_RATIO:
        curr_dir = "./out/ratio0-1/"
        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)

        res = copy.deepcopy(empty_res)

        for ratio in np.arange(0, 1.1, 0.1):
            print "{0} ratio: {1} {0}".format("=" * 5, ratio)
            sub_dir = "{0}/ratio{1}".format(curr_dir, ratio)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)

            num_circle_data = int(default_param["total_num"] * ratio)
            num_noisy_data = int(default_param["total_num"] * (1 - ratio))
            run_ransac_10_times(num_circle_data, default_param["circle_noise"],
                                num_noisy_data, default_param["noisy_noise"], res,
                                save_dir=sub_dir)
        plot_result(res, np.arange(0, 1.1, 0.1),
                    "Circle / Noise Ratio", curr_dir)

    # total number of data points
    # TODO: Change num iterations to adjust for large sample size
    if RUN_TOTAL_NUM:
        curr_dir = "./out/total_num100-10000/"
        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)

        res = copy.deepcopy(empty_res)

        for total_num in np.arange(100, 10200, 1000):
            print "{0} total_num: {1} {0}".format("=" * 5, total_num)
            sub_dir = "{0}/total_num{1}".format(curr_dir, total_num)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)

            num_circle_data = int(total_num * default_param["ratio"])
            num_noisy_data = int(total_num * (1 - default_param["ratio"]))
            run_ransac_10_times(num_circle_data, default_param["circle_noise"],
                                num_noisy_data, default_param["noisy_noise"], res,
                                save_dir=sub_dir)
        plot_result(res, np.arange(100, 10200, 1000),
                    "Total Number Data", curr_dir)

    # circle noise
    if RUN_CIRCLE_NOISE:
        curr_dir = "./out/circle_noise0-1/"
        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)

        res = copy.deepcopy(empty_res)

        for circle_noise in np.arange(0.2, 1.2, 0.2):
            print "{0} circle_noise: {1} {0}".format("=" * 5, circle_noise)
            sub_dir = "{0}/circle_noise{1}".format(curr_dir, circle_noise)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)

            num_circle_data = int(
                default_param["total_num"] * default_param["ratio"])
            num_noisy_data = int(
                default_param["total_num"] * (1 - default_param["ratio"]))
            run_ransac_10_times(num_circle_data, circle_noise,
                                num_noisy_data, default_param["noisy_noise"], res,
                                save_dir=sub_dir)
        plot_result(res, np.arange(0.2, 1.2, 0.2),
                    "Circle Noise", curr_dir)

    # noisy noise
    if RUN_NOISY_NOISE:
        curr_dir = "./out/noisy_noise0-2/"
        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)

        res = copy.deepcopy(empty_res)

        for noisy_noise in np.arange(0.5, 2.5, 0.5):
            print "{0} noisy_noise: {1} {0}".format("=" * 5, noisy_noise)
            sub_dir = "{0}/noisy_noise{1}".format(curr_dir, noisy_noise)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)

            num_circle_data = int(
                default_param["total_num"] * default_param["ratio"])
            num_noisy_data = int(
                default_param["total_num"] * (1 - default_param["ratio"]))
            run_ransac_10_times(num_circle_data, default_param["circle_noise"],
                                num_noisy_data, noisy_noise, res,
                                save_dir=sub_dir)
        plot_result(res, np.arange(0.5, 2.5, 0.5),
                    "Noisy Noise", curr_dir)

    # num iters
    if RUN_NUM_ITERS:
        curr_dir = "./out/num_iters1-200/"
        if not os.path.exists(curr_dir):
            os.mkdir(curr_dir)

        res = copy.deepcopy(empty_res)

        for num_iters in np.arange(1, 220, 20):
            print "{0} num_iters: {1} {0}".format("=" * 5, num_iters)
            sub_dir = "{0}/num_iters{1}".format(curr_dir, num_iters)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)

            num_circle_data = int(
                default_param["total_num"] * default_param["ratio"])
            num_noisy_data = int(
                default_param["total_num"] * (1 - default_param["ratio"]))
            run_ransac_10_times(num_circle_data, default_param["circle_noise"],
                                num_noisy_data, default_param["noisy_noise"], res,
                                num_iters=num_iters, save_dir=sub_dir)
        plot_result(res, np.arange(1, 220, 20),
                    "Iterations", curr_dir)
