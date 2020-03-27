import numpy as np
import numpy.random
import time


class RANSAC:
    """
    A RANSAC Algorithm Class

    Args:
        x_data (np.ndarray): a list of points' x coordinates
        y_data (np.ndarray): a list of points' y coordinates
        n (int): maximum number of iterations to run
        baysac (bool): true if use BAYSAC, otherwise use RANSAC
        d_min (int): the minimum distance learned from the algorithm
        best_model (Tuple[int]): the best model so far
        bayes_probs (np.ndarray): a list of likelihoods that the point at
            index i is an inlier
        bayes_idx (np.ndarray): a list of indices that match the likelihoods
            to the point coordinates
    """

    def __init__(self, x_data, y_data, n, baysac=False):
        assert x_data.shape == y_data.shape, \
            "x_data: {0}, y_data: {1}".format(x_data.shape, y_data.shape)
        self.x_data = x_data
        self.y_data = y_data
        self.n = n
        self.baysac = baysac
        self.d_min = 99999
        self.best_model = None
        self.bayes_probs = np.repeat(0.5, x_data.shape[0])
        self.bayes_idx = np.arange(x_data.shape[0])
        self.s_in = set()

    def sample_points(self):
        """
        Sample 3 points according to self.baysac

        Returns:
            List[Tuple[float]]: The 3 sampled points
        """
        if self.baysac:
            return self._bayes_sampling()
        else:
            return self._random_sampling()

    def _random_sampling(self):
        """
        Randomly sample 3 points.
        Used when self.baysac is false.

        Returns:
            List[Tuple[float]]: The 3 sampled points using random sampling
        """
        sample = []
        save_idx = []
        count = 0

        # get three points from data
        while count < 3:
            curr_idx = np.random.randint(len(self.x_data))

            if curr_idx not in save_idx:
                sample.append((self.x_data[curr_idx], self.y_data[curr_idx]))
                save_idx.append(curr_idx)
                count += 1

        return sample

    def _bayes_sampling(self):
        """
        Sample 3 points with highest likelihoods to be inliers.
        Used when self.baysac is True.

        Returns:
            List[Tuple[float]]: The 3 sampled points using Bayes sampling
        """
        sample = []
        self.bayes_idx = np.argsort(self.bayes_probs)

        for i in range(3):
            # TODO: randomize max
            curr_idx = self.bayes_idx[i]

            sample.append((self.x_data[curr_idx], self.y_data[curr_idx]))

        return sample

    def make_model(self, sample):
        """
        Fit a circle using the 3 sample points

        Args:
            sample (List[Tuple[float]]): the 3 sample points' coordinates

        Returns:
            Tuple[float]: the centre points' x, y coord and the radius
        """
        pt1 = sample[0]
        pt2 = sample[1]
        pt3 = sample[2]

        A = np.array([[pt2[0] - pt1[0], pt2[1] - pt1[1]],
                      [pt3[0] - pt2[0], pt3[1] - pt2[1]]])
        B = np.array([[pt2[0]**2 - pt1[0]**2 + pt2[1]**2 - pt1[1]**2],
                      [pt3[0]**2 - pt2[0]**2 + pt3[1]**2 - pt2[1]**2]])
        inv_A = np.linalg.inv(A)

        c_x, c_y = np.dot(inv_A, B) / 2
        c_x, c_y = c_x[0], c_y[0]
        r = np.sqrt((c_x - pt1[0])**2 + (c_y - pt1[1])**2)

        return c_x, c_y, r

    def eval_model_old(self, model):
        """
        Evaluates the model and calculates the total difference of each point
        being away from the data

        Args:
            model (Tuple[float]): the centre points' x, y coord and the radius

        Returns:
            float: the total difference of each point being away from the data
        """
        d = 0
        c_x, c_y, r = model

        for i in range(len(self.x_data)):
            dis = np.sqrt((self.x_data[i] - c_x) ** 2 +
                          (self.y_data[i] - c_y) ** 2)

            if dis >= r:
                d += dis - r
            else:
                d += r - dis

        return d

    def eval_model(self, model):
        """
        Evaluates the model and calculates the total difference of each point
        being away from the data

        Args:
            model (Tuple[float]): the centre points' x, y coord and the radius

        Returns:
            float: the total difference of each point being away from the data
        """
        d = 0
        c_x, c_y, r = model

        for i in range(len(self.x_data)):
            dis = np.sqrt((self.x_data[i] - c_x) ** 2 +
                          (self.y_data[i] - c_y) ** 2)

            if dis >= r:
                d += dis - r
            else:
                d += r - dis

        return d

    def bayes_eval_model(self, model):
        d = 0
        c_x, c_y, r = model

        for i in range(len(self.x_data)):
            dis = np.sqrt((self.x_data[i] - c_x) ** 2 +
                          (self.y_data[i] - c_y) ** 2)

            if dis >= r:
                d += dis - r
            else:
                d += r - dis

        return d

    def execute_ransac(self):
        # find best model
        start_time = time.time()
        for i in range(self.n):
            curr_sample = self.sample_points()
            model = self.make_model(curr_sample)
            d_temp = self.eval_model(model)

            if self.d_min > d_temp:
                self.best_model = model
                self.d_min = d_temp
        end_time = time.time()
        print "time elapsed: {0}".format(end_time - start_time)
