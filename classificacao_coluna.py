import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def load_data(filepath):
    data = np.genfromtxt(filepath, delimiter=",", dtype="U20", encoding="utf-8")
    X = data[:, :6].astype(float)
    y = data[:, 6]
    return X, y

def one_hot_encode_labels(y):
    N = len(y)
    Y = np.zeros((N, 3))
    for i, label in enumerate(y):
        lab = str(label).strip().upper()
        if lab == "NO":
            Y[i] = [ 1, -1, -1]
        elif lab == "DH":
            Y[i] = [-1,  1, -1]
        elif lab == "SL":
            Y[i] = [-1, -1,  1]
        else:
            Y[i] = [-1, -1, -1]
    return Y

def one_hot_to_index(Y):
    return np.argmax(Y, axis=1)

def compute_confusion_metrics_multi(y_true, y_pred):
    C = 3
    conf = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf[t, p] += 1
    total = len(y_true)
    sensitivities = []
    specificities = []
    for i in range(C):
        TP = conf[i, i]
        FN = np.sum(conf[i, :]) - TP
        FP = np.sum(conf[:, i]) - TP
        TN = total - TP - FN - FP
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        sensitivities.append(sens)
        specificities.append(spec)
    accuracy = np.trace(conf) / total
    return accuracy, np.mean(sensitivities), np.mean(specificities), conf

class ADALINEClassifier:
    def __init__(self, eta, max_epochs, epsilon, random_state=None):
        self.eta = eta
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.random_state = random_state
        self.weights = None
        self.loss_curve = []

    def _compute_eqm(self, X_bias, Y):
        N = X_bias.shape[0]
        total = 0.0
        for i in range(N):
            u = X_bias[i] @ self.weights
            total += np.sum((Y[i] - u)**2)
        return total / (2.0 * N)

    def fit(self, X, Y):
        N = X.shape[0]
        X_bias = np.hstack((np.ones((N, 1)), X))
        C = Y.shape[1]
        self.weights = np.zeros((X_bias.shape[1], C))
        eqm_prev = self._compute_eqm(X_bias, Y)
        for epoch in range(self.max_epochs):
            for i in range(N):
                u = X_bias[i] @ self.weights
                self.weights += self.eta * np.outer(X_bias[i], (Y[i] - u))
            eqm_current = self._compute_eqm(X_bias, Y)
            self.loss_curve.append(eqm_current)
            if abs(eqm_current - eqm_prev) <= self.epsilon:
                break
            eqm_prev = eqm_current
        return self

    def predict(self, X):
        N = X.shape[0]
        X_bias = np.hstack((np.ones((N, 1)), X))
        U = X_bias @ self.weights
        return np.argmax(U, axis=1)

class SimpleMLP:
    def __init__(self, input_dim, hidden_layers, output_dim,
                 eta, max_epochs, epsilon, activation_name="tanh", random_state=None):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.eta = eta
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.activation_name = activation_name
        self.random_state = random_state
        self.weights = []
        self._initialize_weights()
        self.loss_curve = []

    def _initialize_weights(self):
        rgen = np.random.RandomState(self.random_state)
        W0 = rgen.uniform(-0.5, 0.5, (self.hidden_layers[0], self.input_dim + 1))
        self.weights.append(W0)
        for i in range(1, len(self.hidden_layers)):
            Wi = rgen.uniform(-0.5, 0.5, (self.hidden_layers[i], self.hidden_layers[i-1] + 1))
            self.weights.append(Wi)
        inp_out = (self.hidden_layers[-1] + 1) if self.hidden_layers else (self.input_dim + 1)
        W_out = rgen.uniform(-0.5, 0.5, (self.output_dim, inp_out))
        self.weights.append(W_out)

    def activation(self, z):
        if self.activation_name == "tanh":
            return np.tanh(z)
        elif self.activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        else:
            return z

    def activation_derivative(self, z):
        if self.activation_name == "tanh":
            return 1.0 - np.tanh(z)**2
        elif self.activation_name == "sigmoid":
            sig = 1.0 / (1.0 + np.exp(-z))
            return sig * (1 - sig)
        else:
            return np.ones_like(z)

    def forward(self, x):
        self.i_list = []
        self.y_list = []
        x_bias = np.concatenate(([-1], x))
        for W in self.weights:
            i = W @ x_bias
            self.i_list.append(i)
            y = self.activation(i)
            self.y_list.append(y)
            x_bias = np.concatenate(([-1], y))
        return self.y_list[-1]

    def backward(self, x, d):
        L = len(self.weights) - 1
        delta = [None] * (L + 1)
        delta[L] = self.activation_derivative(self.i_list[L]) * (d - self.y_list[L])
        for j in range(L-1, -1, -1):
            W_next = self.weights[j+1]
            delta[j] = self.activation_derivative(self.i_list[j]) * (W_next[:,1:].T @ delta[j+1])
        x_bias = np.concatenate(([-1], x))
        self.weights[0] += self.eta * np.outer(delta[0], x_bias)
        for j in range(1, len(self.weights)):
            y_bias = np.concatenate(([-1], self.y_list[j-1]))
            self.weights[j] += self.eta * np.outer(delta[j], y_bias)

    def train(self, X, Y):
        N = X.shape[0]
        epoch = 0
        EQM = np.inf
        self.loss_curve = []
        while EQM > self.epsilon and epoch < self.max_epochs:
            soma_erros = 0.0
            for i in range(N):
                out = self.forward(X[i])
                erro = Y[i] - out
                soma_erros += np.sum(erro**2)
                self.backward(X[i], Y[i])
            EQM = soma_erros / (2.0 * N)
            self.loss_curve.append(EQM)
            epoch += 1
        return self

    def predict(self, X):
        return np.array([np.argmax(self.forward(x)) for x in X])

def plot_2d_scatter(X, y):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="coolwarm", edgecolor='k', alpha=0.7)
    plt.xlabel("Incidência Pélvica")
    plt.ylabel("Inclinação Pélvica")
    plt.title("Gráfico de Dispersão 2D das Amostras")
    plt.legend(title="Condição")
    plt.show()

def plot_confusion_matrix(cm, title="Matriz de Confusão"):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Predito NO", "Predito DH", "Predito SL"],
                yticklabels=["Real NO", "Real DH", "Real SL"])
    plt.title(title)
    plt.ylabel("Valor Real")
    plt.xlabel("Valor Predito")
    plt.show()

def plot_learning_curve(curve, title="Curva de Aprendizado"):
    plt.figure(figsize=(8,6))
    plt.plot(curve, marker='o', linestyle='-')
    plt.xlabel("Épocas")
    plt.ylabel("Erro Quadrático Médio")
    plt.title(title)
    plt.grid(True)
    plt.show()

def monte_carlo_classification(X, Y, R=100):
    metrics_adaline, metrics_mlp = [], []
    best_adaline = {"acc": -1, "cm": None, "curve": None}
    worst_adaline = {"acc": 2, "cm": None, "curve": None}
    best_mlp     = {"acc": -1, "cm": None, "curve": None}
    worst_mlp    = {"acc": 2, "cm": None, "curve": None}

    n = X.shape[0]
    for r in range(R):
        idx = np.random.permutation(n)
        t = int(0.8*n)
        Xtr, Xte = X[idx[:t]], X[idx[t:]]
        Ytr, Yte = Y[idx[:t]], Y[idx[t:]]
        yte_idx = one_hot_to_index(Yte)

        ad = ADALINEClassifier(eta=0.001, max_epochs=200, epsilon=1e-4, random_state=r)
        ad.fit(Xtr, Ytr)
        pa = ad.predict(Xte)
        acc_a, sens_a, spec_a, cm_a = compute_confusion_metrics_multi(yte_idx, pa)
        metrics_adaline.append((acc_a, sens_a, spec_a))
        if acc_a > best_adaline["acc"]:
            best_adaline.update(acc=acc_a, cm=cm_a, curve=ad.loss_curve.copy())
        if acc_a < worst_adaline["acc"]:
            worst_adaline.update(acc=acc_a, cm=cm_a, curve=ad.loss_curve.copy())

        mlp = SimpleMLP(X.shape[1], [10], 3, eta=0.1, max_epochs=300, epsilon=1e-4,
                        activation_name="tanh", random_state=r)
        mlp.train(Xtr, Ytr)
        pm = mlp.predict(Xte)
        acc_m, sens_m, spec_m, cm_m = compute_confusion_metrics_multi(yte_idx, pm)
        metrics_mlp.append((acc_m, sens_m, spec_m))
        if acc_m > best_mlp["acc"]:
            best_mlp.update(acc=acc_m, cm=cm_m, curve=mlp.loss_curve.copy())
        if acc_m < worst_mlp["acc"]:
            worst_mlp.update(acc=acc_m, cm=cm_m, curve=mlp.loss_curve.copy())

        sys.stdout.write(f"\rProgresso Monte Carlo: {100*(r+1)/R:5.1f}%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return metrics_adaline, metrics_mlp, best_adaline, worst_adaline, best_mlp, worst_mlp

def print_summary(name, mets):
    mets = np.array(mets)
    mean, std, mx, mn = np.mean(mets,0), np.std(mets,0), np.max(mets,0), np.min(mets,0)
    print(f"\n{name}:")
    print(f"{'Métrica':<12}{'Média':>8}{'Std':>8}{'Máx':>8}{'Mín':>8}")
    labels = ["Acc", "Sens", "Spec"]
    for i, l in enumerate(labels):
        print(f"{l:<12}{mean[i]:8.4f}{std[i]:8.4f}{mx[i]:8.4f}{mn[i]:8.4f}")

def experiment_topologies(Xtr, Ytr, Xte, Yte):
    yte_idx = one_hot_to_index(Yte)
    under = SimpleMLP(Xtr.shape[1], [1], 3, 0.1, 200, 1e-4, "tanh", 42).train(Xtr, Ytr)
    pu = under.predict(Xte)
    au, su, spu, cmu = compute_confusion_metrics_multi(yte_idx, pu)
    print(f"\nUnderfitting MLP: Acurácia={au:.4f}, Sensibilidade={su:.4f}, Especificidade={spu:.4f}")
    plot_confusion_matrix(cmu, "Matriz de Confusão – Underfitting")
    plot_learning_curve(under.loss_curve, "Curva de Aprendizado – Underfitting")

    over = SimpleMLP(Xtr.shape[1], [50,50], 3, 0.1, 200, 1e-4, "tanh", 42).train(Xtr, Ytr)
    po = over.predict(Xte)
    ao, so, spo, cmo = compute_confusion_metrics_multi(yte_idx, po)
    print(f"\nOverfitting MLP: Acurácia={ao:.4f}, Sensibilidade={so:.4f}, Especificidade={spo:.4f}")
    plot_confusion_matrix(cmo, "Matriz de Confusão – Overfitting")
    plot_learning_curve(over.loss_curve, "Curva de Aprendizado – Overfitting")

def main():
    filepath = r"dados/coluna_vertebral.csv"
    X, y = load_data(filepath)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    Y = one_hot_encode_labels(y)

    plot_2d_scatter(X, y)
    ad_mets, mlp_mets, bA, wA, bM, wM = monte_carlo_classification(X, Y, R=100)
    print_summary("ADALINE", ad_mets)
    print_summary("MLP", mlp_mets)

    print(f"\n→ ADALINE melhor: Acurácia={bA['acc']:.4f}")
    plot_confusion_matrix(bA["cm"], "Matriz de Confusão – ADALINE Melhor")
    plot_learning_curve(bA["curve"], "Curva – ADALINE Melhor")
    print(f"\n→ ADALINE pior: Acurácia={wA['acc']:.4f}")
    plot_confusion_matrix(wA["cm"], "Matriz de Confusão – ADALINE Pior")
    plot_learning_curve(wA["curve"], "Curva – ADALINE Pior")

    print(f"\n→ MLP melhor: Acurácia={bM['acc']:.4f}")
    plot_confusion_matrix(bM["cm"], "Matriz de Confusão – MLP Melhor")
    plot_learning_curve(bM["curve"], "Curva – MLP Melhor")
    print(f"\n→ MLP pior: Acurácia={wM['acc']:.4f}")
    plot_confusion_matrix(wM["cm"], "Matriz de Confusão – MLP Pior")
    plot_learning_curve(wM["curve"], "Curva – MLP Pior")

    np.random.seed(42)
    idx = np.random.permutation(X.shape[0])
    t = int(0.8 * X.shape[0])
    experiment_topologies(X[idx[:t]], Y[idx[:t]], X[idx[t:]], Y[idx[t:]])

if __name__ == "__main__":
    main()
