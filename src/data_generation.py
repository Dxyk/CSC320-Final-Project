import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import sklearn.datasets


def generate_circle(num_samples=100, noise=0.1):
    """
    Generates the x, y coordinates of noisy points scattered around a circle

    Args:
        num_samples (int, optional): the number of samples. Defaults to 100.
        noise (float, optional): the variance of the noise. Defaults to 0.1.

    Returns:
        Tuple[np.ndarray]: the x, y coordinates of the circle
    """
    data, label = sklearn.datasets.make_circles(n_samples=2 * num_samples,
                                                shuffle=False,
                                                noise=noise,
                                                random_state=1,
                                                factor=0.8)
    circle_data = data[label == 1]
    x = circle_data[:, 0]
    y = circle_data[:, 1]
    return x, y


def generate_noise(num_samples=20, noise=1, x_bound=1.5, y_bound=1.5):
    """
    Generate the x, y coordinates of normal noisy points

    Args:
        num_samples (int, optional): the number of samples. Defaults to 20.
        noise (int, optional): the variance of the normal noise. Defaults to 1.
        x_bound (float, optional): the radius of the x axis. Defaults to 1.5.
        y_bound (float, optional): the radius of the y axis. Defaults to 1.5.

    Returns:
        Tuple[np.ndarray]: the x, y coordinates of the points
    """
    x_noise = np.random.randn(num_samples) * noise
    y_noise = np.random.randn(num_samples) * noise
    return x_noise, y_noise


if __name__ == "__main__":
    num_circle_sample = 100
    circle_noise = 0.1
    num_noisy_sample = 20
    noisy_noise = 1
    x_bound = 1.5
    y_bound = 1.5

    x_circle, y_circle = generate_circle(
        num_samples=num_circle_sample,
        noise=circle_noise
    )
    x_noise, y_noise = generate_noise(
        num_samples=num_noisy_sample,
        noise=noisy_noise,
        x_bound=x_bound,
        y_bound=y_bound
    )

    x = np.append(x_circle, x_noise)
    y = np.append(y_circle, y_noise)

    plt.scatter(x, y, c="blue", marker=".", label="data")
    plt.axis("scaled")
    plt.show()
    plt.clf()
