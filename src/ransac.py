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
        is_baysac (bool): true if use BAYSAC, otherwise use RANSAC
        min_dist (int): the minimum distance learned by the algorithm
        best_model (Tuple[float]): the best model so far
        likelihoods (np.ndarray): a list of likelihoods that the point at
            index i is an inlier
        bayes_idx (np.ndarray): a list of indices that match the likelihoods
            to the point coordinates
        inliers (Set[Tuple[float]]): the coordinates that are inliers
        runtime (float): the runtime for execute_ransac
    """

    def __init__(self, x_data, y_data, n, threshold, is_baysac=False):
        assert x_data.shape == y_data.shape, "x and y shapes must match." + \
            "x_data: {0}, y_data: {1}.".format(x_data.shape, y_data.shape)
        assert x_data.shape[0] >= 3, "there must be at least 3 data " + \
            "points to fit a circle. Given {0}.".format(x_data.shape[0])

        # TODO: we may want to calculate n instead of defining it
        self.x_data = x_data
        self.y_data = y_data
        self.n = n
        self.threshold = threshold
        self.is_baysac = is_baysac
        self.min_dist = float("inf")
        self.best_model = (0., 0., 0.)
        self.likelihoods = np.repeat(0.5, x_data.shape[0])
        self.inliers = set()
        self.runtime = 0.

    def sample_indices(self):
        """
        Sample 3 points' indices.

        Returns:
            List[int]: 3 points' indices
        """
        if self.is_baysac:
            return self._likelihood_sampling()
        else:
            return self._random_sampling()

    def _random_sampling(self):
        """
        Sample 3 points' indices using random sampling.
        Used when self.is_baysac is false.

        Returns:
            List[int]: 3 points' indices using random samping
        """
        indices = np.indices(self.x_data.shape)[0]
        sample_indices = np.random.choice(indices, 3, replace=False)
        return sample_indices

    def _likelihood_sampling(self):
        """
        Sample 3 points' indices using highest likelihoods.
        Used when self.is_baysac is True.

        Returns:
            List[int]: 3 points' indices using with maximum likelihood
        """
        # sort in decreasing order
        indices = np.argsort(self.likelihoods)[::-1]
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
        curr_dist_total = 0

        for i in range(len(self.x_data)):
            curr_x = self.x_data[i]
            curr_y = self.y_data[i]

            dist = abs(np.sqrt((curr_x - c_x) ** 2 + (curr_y - c_y) ** 2) - r)

            if dist < self.threshold:
                inliers.add((curr_x, curr_y))

            curr_dist_total += dist

        if len(inliers) > len(self.inliers):
            self.inliers = inliers
            self.best_model = model
            self.min_dist = curr_dist_total
        return

    def update_likelihoods(self, sample_indices):
        """
        Update the likelihoods given the current sample indices

        Args:
            sample_indices (List[int]): the 3 sample points' indices
        """
        curr_likelihoods = self.likelihoods[sample_indices]
        p_sample_subset_inlier = np.prod(curr_likelihoods)
        self.likelihoods[sample_indices] = (curr_likelihoods - p_sample_subset_inlier) / \
            (1 - p_sample_subset_inlier)
        return

    def execute_ransac(self):
        """
        The top level method for executing ransac algorithm
        """
        start_time = time.time()
        for i in range(self.n):
            curr_sample_indices = self.sample_indices()
            model = self.make_model(curr_sample_indices)
            self.eval_model(model)
            if self.is_baysac:
                self.update_likelihoods(curr_sample_indices)
        end_time = time.time()
        self.runtime = end_time - start_time
        return

    def get_best_model(self):
        """
        Get the best model

        Returns:
            Tuple[float]: the best model
        """
        return self.best_model

    def get_inliers(self):
        """
        Get the set of inlier points

        Returns:
            Set[Tuple[Float]]: the set of inlier points
        """
        return self.inliers

    def get_total_dist(self):
        """
        Get the total distance from all points to the circle

        Returns:
            float: the total distance from all points to the circle
        """
        return self.min_dist

    def get_inlier_dist(self):
        """
        Get the total distance from all inlier points to the circle

        Returns:
            float: the total distance from all inliers to the circle
        """
        c_x, c_y, r = self.best_model
        x_inlier, y_inlier = zip(*list(self.inliers))
        total_dist = 0
        for i in range(len(self.inliers)):
            curr_x = x_inlier[i]
            curr_y = y_inlier[i]

            dist = abs(np.sqrt((curr_x - c_x) ** 2 + (curr_y - c_y) ** 2) - r)

            total_dist += dist
        return total_dist

    def get_runtime(self):
        """
        Get the runtime for execute_ransac method

        Returns:
            float: the runtime for execute_ransac method
        """
        return self.runtime
