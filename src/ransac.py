import numpy as np
import numpy.random
import time


class Ransac:
    """
    A Ransac Algorithm Class

    Args:
        x_data (np.ndarray): a list of points' x coordinates
        y_data (np.ndarray): a list of points' y coordinates
        n (int): maximum number of iterations to run
        threshold (float): the threshold to determine if a points is an inlier
        baysac (bool): true if use BAYSAC, otherwise use RANSAC
        d_min (int): the minimum distance learned from the algorithm
        best_model (Tuple[int]): the best model so far
        likelihoods (np.ndarray): a list of likelihoods that the point at
            index i is an inlier
        bayes_idx (np.ndarray): a list of indices that match the likelihoods
            to the point coordinates
        inliers (Set[Tuple[float]]): the coordinates that are inliers
    """

    def __init__(self, x_data, y_data, n, threshold, baysac=False):
        assert x_data.shape == y_data.shape, "x and y shapes must match." + \
            "x_data: {0}, y_data: {1}.".format(x_data.shape, y_data.shape)
        assert x_data.shape[0] >= 3, "there must be at least 3 data " + \
            "points to fit a circle. Given {0}.".format(x_data.shape[0])

        # TODO: we may want to calculate n instead of defining it
        self.x_data = x_data
        self.y_data = y_data
        self.n = n
        self.threshold = threshold
        self.baysac = baysac
        self.d_min = 99999
        self.best_model = None
        self.likelihoods = np.repeat(0.5, x_data.shape[0])
        self.inliers = set()

    def sample_indices(self):
        """
        Sample 3 points' indices.

        Returns:
            List[int]: 3 points' indices
        """
        if self.baysac:
            return self._likelihood_sampling()
        else:
            return self._random_sampling()

    def _random_sampling(self):
        """
        Sample 3 points' indices using random sampling.
        Used when self.baysac is false.

        Returns:
            List[int]: 3 points' indices using random samping
        """
        indices = np.indices(self.x_data.shape)[0]
        sample_indices = np.random.choice(indices, 3, replace=False)
        return sample_indices
        # sample = []
        # save_idx = []
        # count = 0

        # # get three points from data
        # while count < 3:
        #     curr_idx = np.random.randint(len(self.x_data))

        #     if curr_idx not in sample:
        #         sample.append(curr_idx)
        #         count += 1

        # return sample

    def _likelihood_sampling(self):
        """
        Sample 3 points' indices using highest likelihoods.
        Used when self.baysac is True.

        Returns:
            List[int]: 3 points' indices using with maximum likelihood
        """
        indices = np.argsort(self.likelihoods)
        sample_indices = indices[:3]

        return sample_indices

    def make_model(self, sample_indices):
        """
        Fit a circle using the 3 sample points

        Args:
            sample_indices (List[int]): the 3 sample points' indices

        Returns:
            Tuple[float]: the centre points' x, y coord and the radius
        """
        first_three_indices = sample_indices[:3]

        pt1, pt2, pt3 = zip(self.x_data[first_three_indices],
                            self.y_data[first_three_indices])

        A = np.array([[pt2[0] - pt1[0], pt2[1] - pt1[1]],
                      [pt3[0] - pt2[0], pt3[1] - pt2[1]]])
        B = np.array([[pt2[0]**2 - pt1[0]**2 + pt2[1]**2 - pt1[1]**2],
                      [pt3[0]**2 - pt2[0]**2 + pt3[1]**2 - pt2[1]**2]])
        inv_A = np.linalg.inv(A)

        c_x, c_y = np.dot(inv_A, B) / 2
        c_x, c_y = c_x[0], c_y[0]
        r = np.sqrt((c_x - pt1[0])**2 + (c_y - pt1[1])**2)

        return c_x, c_y, r

    def eval_model(self, model):
        """
        Evaluates the model and calculates the total difference of each point
        being away from the data

        Args:
            model (Tuple[float]): the centre points' x, y coord and the radius

        Returns:
            float: the total difference of each point being away from the data
        """
        c_x, c_y, r = model
        inliers = set()

        for i in range(len(self.x_data)):
            curr_x = self.x_data[i]
            curr_y = self.y_data[i]

            dist = np.sqrt((self.x_data[i] - c_x) ** 2 +
                           (self.y_data[i] - c_y) ** 2) - r

            if abs(dist) < self.threshold:
                inliers.add((curr_x, curr_y))

        if len(inliers) > len(self.inliers):
            self.inliers = inliers
            self.best_model = model
        return

    def update_likelihoods(self, sample_indices):
        # P(H_t not subset of Inliers) = 1 - P(H_t subset of Inliers)
        curr_likelihoods = self.likelihoods[sample_indices]
        # p_sample_not_inlier = 1 - np.prod()
        return

    def execute_ransac(self):
        start_time = time.time()
        for i in range(self.n):
            curr_sample_indices = self.sample_indices()
            model = self.make_model(curr_sample_indices)
            self.eval_model(model)
            if self.baysac:
                self.update_likelihoods(curr_sample_indices)
        end_time = time.time()
        print "time elapsed: {0}".format(end_time - start_time)
