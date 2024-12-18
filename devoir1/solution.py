import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d


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
        one_hot_labels = np.zeros((len(self.train_targets), self.labels.max() + 1))
        one_hot_labels[np.arange(len(self.train_targets)), self.train_targets] = 1
        for index, test_input in enumerate(test_data):
            rbf_kernel_vals = np.exp(
                -0.5
                * np.sum(np.abs(test_input - self.train_inputs), axis=1) ** 2
                / self.sigma**2
            )
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
    error_rate_val = ErrorRate(train[:, :-1], train[:, -1], val[:, :-1], val[:, -1])
    h_opt = params[np.argmin([error_rate_val.hard_parzen(param) for param in params])]
    sig_opt = params[np.argmin([error_rate_val.soft_parzen(param) for param in params])]
    error_rate_test = ErrorRate(train[:, :-1], train[:, -1], test[:, :-1], test[:, -1])
    return [error_rate_test.hard_parzen(h_opt), error_rate_test.soft_parzen(sig_opt)]


def random_projections(X, A):
    return (1 / np.sqrt(2)) * X @ A


params = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]

iris = np.genfromtxt("iris.txt")

train, val, test = split_dataset(iris)
error_rate = ErrorRate(train[:, :-1], train[:, -1], val[:, :-1], val[:, -1])

hard_parzen_error_rates = [error_rate.hard_parzen(param) for param in params]
soft_rbf_parzen_error_rate = [error_rate.soft_parzen(param) for param in params]

plt.plot(params, hard_parzen_error_rates, marker="o", label="hard parzen")
plt.plot(params, soft_rbf_parzen_error_rate, marker="o", label="soft rbf parzen")

plt.xlabel("h \\ sigma")
plt.ylabel("taux d'erreur")
plt.title("taux d'erreur de validation")

plt.legend(loc="upper right")
plt.grid()
plt.savefig("error_rate.png")

plt.show()

proj_matrices = np.random.randn(500, 4, 2)
train_proj = [random_projections(train[:, :-1], proj_mat) for proj_mat in proj_matrices]
val_proj = [random_projections(val[:, :-1], proj_mat) for proj_mat in proj_matrices]

error_rate_proj = [
    ErrorRate(train_proj[index], train[:, -1], val_proj[index], val[:, -1])
    for index in range(500)
]

error_rate_hard_val_proj = [
    [error_rate.hard_parzen(param) for param in params]
    for error_rate in error_rate_proj
]

error_rate_soft_val_proj = [
    [error_rate.soft_parzen(param) for param in params]
    for error_rate in error_rate_proj
]

plt.errorbar(
    params,
    0.002 * np.sum(error_rate_hard_val_proj, axis=0),
    yerr=0.2 * np.std(error_rate_hard_val_proj, axis=0),
    marker="o",
    label="hard parzen",
    capsize=4,
)

plt.errorbar(
    params,
    0.002 * np.sum(error_rate_soft_val_proj, axis=0),
    yerr=0.2 * np.std(error_rate_soft_val_proj, axis=0),
    marker="o",
    label="soft parzen",
    capsize=4,
)
plt.xlabel("h \\ sigma")
plt.ylabel("taux d'erreur moyen")
plt.title("taux d'erreur moyens de validation pour chaque valeur de h et sigma")

plt.legend(loc="upper right")
plt.grid()

plt.savefig("mean_error_rate.png")
plt.show()

points = np.array([[5, -2], [2, -10], [2, -15], [8, 0], [2, 5], [8, 5]])
vor = Voronoi(points)
voronoi_plot_2d(
    vor, show_vertices=False, line_colors="blue", line_width=2, line_alpha=0.6
)
plt.plot(points[:3, 0], points[:3, 1], "ro", label="Classe 1")
plt.plot(points[3:, 0], points[3:, 1], "go", label="Classe 2")
plt.title("Diagramme de Voronoi avec 2 classes")
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid()
plt.savefig("voronoi_diagram_2_class.png")
plt.show()
mean_class_1 = np.mean(points[:3], axis=0)
mean_class_2 = np.mean(points[3:], axis=0)

plt.plot(points[:3, 0], points[:3, 1], "ro", label="Classe 1")
plt.plot(points[3:, 0], points[3:, 1], "go", label="Classe 2")

plt.plot(
    mean_class_1[0],
    mean_class_1[1],
    color="r",
    marker="X",
    label="Moyenne Classe 0",
)
plt.plot(
    mean_class_2[0],
    mean_class_2[1],
    color="g",
    marker="X",
    label="Moyenne Classe 1",
)

decision_boundary = (mean_class_1 + mean_class_2) / 2
plt.axline(
    decision_boundary,
    slope=-(mean_class_2[0] - mean_class_1[0]) / (mean_class_2[1] - mean_class_1[1]),
    color="black",
    linestyle="--",
    label="Frontière de décision",
)
plt.title("Classification binaire")
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid()
plt.savefig("binary_classification.png")
plt.show()

points = np.array(
    [[5, -2], [2, -10], [2, -15], [8, 0], [2, 5], [8, 5], [3, 1], [-3, 0], [5, -8]]
)
vor = Voronoi(points)
voronoi_plot_2d(
    vor, show_vertices=False, line_colors="blue", line_width=2, line_alpha=0.6
)
plt.plot(points[:3, 0], points[:3, 1], "ro", label="Classe 1")
plt.plot(points[3:6, 0], points[3:6, 1], "go", label="Classe 2")
plt.plot(points[6:, 0], points[6:, 1], "bo", label="Classe 3")
plt.title("Diagramme de Voronoi avec 3 classes")
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid()
plt.savefig("voronoi_diagram_3_class.png")
plt.show()
mean_class_1 = np.mean(points[:3], axis=0)
mean_class_2 = np.mean(points[3:6], axis=0)
mean_class_3 = np.mean(points[6:], axis=0)

vor = Voronoi([mean_class_1, mean_class_2, mean_class_3])
voronoi_plot_2d(
    vor, show_vertices=False, line_colors="blue", line_width=2, line_alpha=0.6
)
plt.plot(
    mean_class_1[0],
    mean_class_1[1],
    color="r",
    marker="X",
    label="Moyenne Classe 0",
)
plt.plot(
    mean_class_2[0],
    mean_class_2[1],
    color="g",
    marker="X",
    label="Moyenne Classe 1",
)
plt.plot(
    mean_class_3[0],
    mean_class_3[1],
    color="b",
    marker="X",
    label="Moyenne Classe 2",
)
plt.plot(points[:3, 0], points[:3, 1], "ro", label="Classe 1")
plt.plot(points[3:6, 0], points[3:6, 1], "go", label="Classe 2")
plt.plot(points[6:, 0], points[6:, 1], "bo", label="Classe 3")

plt.title("Classification multiple")
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid()
plt.savefig("multi_classification.png")
plt.show()
