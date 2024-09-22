import matplotlib.pyplot as plt
import numpy as np


######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)


#############################################


class Q1:
    def feature_means(self, iris):
        return np.mean(iris[:, :-1], axis=0)

    def empirical_covariance(self, iris):
        return np.cov(np.transpose(iris[:, :-1]))

    def feature_means_class_1(self, iris):
        return np.mean(iris[iris[:, -1] == 1, :-1], axis=0)

    def empirical_covariance_class_1(self, iris):
        return np.cov(np.transpose(iris[iris[:, -1] == 1, :-1]))


class HardParzen:
    def __init__(self, h):
        self.h = h
        self.labels = None
        self.train_inputs = None
        self.train_targets = None

    def fit(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_targets = np.array(train_labels, dtype=np.int8)
        self.labels = np.array(np.unique(train_labels), dtype=np.int8)

    def predict(self, test_data):
        Y_pred = np.zeros(len(test_data))
        for index, test_input in enumerate(test_data):
            distances = np.sum(np.abs(test_input - self.train_inputs), axis=1)
            neighbors = self.train_targets[distances < self.h]
            if neighbors.size == 0:
                Y_pred[index] = draw_rand_label(test_input, self.labels)
            else:
                Y_pred[index] = np.argmax(np.bincount(neighbors))
        return Y_pred


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma = sigma
        self.labels = None
        self.train_inputs = None
        self.train_targets = None

    def fit(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_targets = np.array(train_labels, dtype=np.int8)
        self.labels = np.array(np.unique(train_labels), dtype=np.int8)

    def predict(self, test_data):
        Y_pred = np.zeros(len(test_data))
        for index, test_input in enumerate(test_data):
            rbf_kernel_vals = np.exp(
                -0.5
                * np.sum(np.abs(test_input - self.train_inputs), axis=1) ** 2
                / self.sigma**2
            )
            one_hot_labels = np.zeros((len(self.train_targets), self.labels.max() + 1))
            one_hot_labels[np.arange(len(self.train_targets)), self.train_targets] = 1
            Y_pred[index] = np.argmax(
                np.sum(rbf_kernel_vals[:, None] * one_hot_labels, axis=0)
            )
        return Y_pred


def split_dataset(iris):
    train = iris[np.isin(np.arange(len(iris)) % 5, [0, 1, 2])]
    validation = iris[np.arange(len(iris)) % 5 == 3]
    test = iris[np.arange(len(iris)) % 5 == 4]
    return train, validation, test


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        hardParzen = HardParzen(h)
        hardParzen.fit(self.x_train, self.y_train)
        y_pred = hardParzen.predict(self.x_val)
        return float(np.sum(y_pred != self.y_val) / len(self.y_val))

    def soft_parzen(self, sigma):
        softParzen = SoftRBFParzen(sigma)
        softParzen.fit(self.x_train, self.y_train)
        y_pred = softParzen.predict(self.x_val)
        return float(np.sum(y_pred != self.y_val) / len(self.y_val))


def get_test_errors(iris):
    train, val, test = split_dataset(iris)
    e_val = ErrorRate(train[:, :-1], train[:, -1], val[:, :-1], val[:, -1])
    h_opt = params[np.argmin([e_val.hard_parzen(param) for param in params])]
    sig_opt = params[np.argmin([e_val.soft_parzen(param) for param in params])]
    e_test = ErrorRate(train[:, :-1], train[:, -1], test[:, :-1], test[:, -1])
    return [e_test.hard_parzen(h_opt), e_test.soft_parzen(sig_opt)]


def random_projections(X, A):
    pass


iris = np.genfromtxt("iris.txt")

train, val, test = split_dataset(iris)
e = ErrorRate(train[:, :-1], train[:, -1], val[:, :-1], val[:, -1])

params = [0.01, 0.1, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
hard_parzen_error_rates = [e.hard_parzen(param) for param in params]
soft_rbf_parzen_error_rate = [e.soft_parzen(param) for param in params]

plt.plot(params, hard_parzen_error_rates, label="hard parzen")
plt.plot(params, soft_rbf_parzen_error_rate, label="soft rbf parzen")

plt.xlabel("paramètre")
plt.ylabel("taux d'erreur")

plt.legend(loc="upper right")

plt.savefig("error_rate.png")

plt.show()
