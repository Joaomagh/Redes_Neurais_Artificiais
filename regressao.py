# ==============================
# 1. Importações
# ==============================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Semente para reprodutibilidade
np.random.seed(42)

# ==============================
# 2. Organização do Conjunto de Dados
# ==============================
def carregar_dados(filepath):
    data = np.loadtxt(filepath)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    return X, y

def normalizar_dados(X, y):
    # Normalização Z-score
    X_norm = (X - X.mean()) / X.std()
    y_norm = (y - y.mean()) / y.std()
    return X_norm, y_norm

# ==============================
# 3. Visualização dos Dados
# ==============================
def plot_dados_originais(X, y):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X.flatten(), y=y.flatten())
    plt.xlabel("Velocidade do vento (original)")
    plt.ylabel("Potência gerada (original)")
    plt.title("Gráfico de Dispersão - Dados Originais")
    plt.grid(True)
    plt.show()

def plot_dados_normalizados(X, y):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=X.flatten(), y=y.flatten())
    plt.xlabel("Velocidade do vento (normalizada)")
    plt.ylabel("Potência gerada (normalizada)")
    plt.title("Gráfico de Dispersão - Dados Normalizados (Z-score)")
    plt.grid(True)
    plt.show()

# ==============================
# 4. Implementação dos Modelos
# ==============================

class Adaline:
    def __init__(self, eta=0.001, epochs=100, tol=1e-3):
        self.eta = eta
        self.epochs = epochs
        self.tol = tol

    def fit(self, X, y):
        self.w = np.zeros((X.shape[1] + 1, 1))
        self.mse_list = []
        for _ in range(self.epochs):
            y_pred = self.net_input(X)
            errors = y - y_pred
            self.w[1:] += self.eta * X.T @ errors
            self.w[0] += self.eta * errors.sum()
            mse = np.mean(errors**2)  # Correção: MSE sem divisão por 2
            self.mse_list.append(mse)
            if mse < self.tol:
                break
        return self

    def net_input(self, X):
        return X @ self.w[1:] + self.w[0]

    def predict(self, X):
        return self.net_input(X)

class MLP:
    def __init__(self, input_size, hidden_layers, output_size=1, lr=0.01, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * 0.01 for i in range(len(self.layers)-1)]
        self.biases = [np.zeros((1, s)) for s in self.layers[1:]]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def fit(self, X, y):
        self.losses = []
        for _ in range(self.epochs):
            a = X
            activations = [a]
            zs = []
            for w, b in zip(self.weights, self.biases):
                z = a @ w + b
                zs.append(z)
                a = self.sigmoid(z)
                activations.append(a)

            delta = (activations[-1] - y) * self.dsigmoid(zs[-1])
            deltas = [delta]
            for l in range(2, len(self.layers)):
                z = zs[-l]
                delta = deltas[-1] @ self.weights[-l+1].T * self.dsigmoid(z)
                deltas.append(delta)
            deltas.reverse()

            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * activations[i].T @ deltas[i]
                self.biases[i] -= self.lr * np.mean(deltas[i], axis=0)

            mse = np.mean((activations[-1] - y)**2)  # Correção: MSE sem divisão por 2
            self.losses.append(mse)

    def predict(self, X):
        a = X
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(a @ w + b)
        return a

# ==============================
# 5. Validação Monte Carlo
# ==============================
def monte_carlo(model_class, model_args, X, y, R=250):
    results = []
    for _ in range(R):
        idx = np.random.permutation(len(X))
        train_idx = idx[:int(0.8 * len(X))]
        test_idx = idx[int(0.8 * len(X)):] 

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = model_class(**model_args)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = np.mean((y_test - y_pred)**2)  # Correção: MSE sem divisão por 2
        results.append(mse)

    results = np.array(results)
    return {
        'Média': results.mean(),
        'Desvio-padrão': results.std(),
        'Máximo': results.max(),
        'Mínimo': results.min()
    }

# ==============================
# 6. Execução principal (main)
# ==============================
if __name__ == "__main__":
    filepath = "aerogerador.dat"
    
    # Carregar dados
    X, y = carregar_dados(filepath)
    
    # Visualização dos dados originais
    plot_dados_originais(X, y)

    # Normalizar os dados com Z-score
    X, y = normalizar_dados(X, y)

    # Visualização dos dados normalizados
    plot_dados_normalizados(X, y)

    print("\nValidação com ADALINE:")
    stats_adaline = monte_carlo(Adaline, {'eta': 0.001, 'epochs': 100}, X, y)
    print(stats_adaline)

    print("\nValidação com MLP:")
    stats_mlp = monte_carlo(MLP, {'input_size': 1, 'hidden_layers': [10], 'epochs': 200}, X, y)
    print(stats_mlp)

    # 🔥 Gráfico da curva de aprendizado 🔥
    mlp_under = MLP(input_size=1, hidden_layers=[2], epochs=200)
    mlp_under.fit(X, y)

    mlp_intermediate = MLP(input_size=1, hidden_layers=[10], epochs=200)
    mlp_intermediate.fit(X, y)

    mlp_over = MLP(input_size=1, hidden_layers=[50, 50, 50], epochs=200)
    mlp_over.fit(X, y)

    plt.figure(figsize=(10, 5))
    plt.plot(mlp_under.losses, label="Subdimensionado (2 neurônios)")
    plt.plot(mlp_intermediate.losses, label="Intermediário (10 neurônios)")
    plt.plot(mlp_over.losses, label="Superdimensionado (camadas=[50,50,50])")
    plt.xlabel("Épocas")
    plt.ylabel("Custo (MSE)")
    plt.title("Curvas de Aprendizado - MLP")
    plt.legend()
    plt.grid(True)
    plt.show()
