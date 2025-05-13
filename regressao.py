import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# ===============================
# FUNÇÃO DE NORMALIZAÇÃO DOS DADOS
# ===============================
def normalize_data(X, y):
    """Normaliza os dados:
         X_norm = (X - média) / desvio-padrão
         y_norm = (y - média) / desvio-padrão
    """
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_norm = (X - X_mean) / X_std

    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std

    return X_norm, y_norm

# ===============================
# 1. IMPLEMENTAÇÃO DO ADALINE
# ===============================
class Adaline:
    def __init__(self, eta=0.01, max_epochs=100, epsilon=1e-4, random_state=1):
        """
        Configuração do ADALINE:
         - Taxa de aprendizagem η = 0.01.
         - Máximo de épocas = 100.
         - Critério de parada ε = 1e-4.
         - Inicializa os pesos (com bias) com zeros.
        """
        self.eta = eta
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.random_state = random_state
        self.weights = None

    def _eqm(self, X, y):
        """Calcula o Erro Quadrático Médio (EQM):
               EQM = (1/(2N)) * Σ (d - u)²
        """
        N = X.shape[0]
        eqm = 0.0
        for i in range(N):
            u = np.dot(self.weights, X[i])
            eqm += (y[i] - u) ** 2
        return eqm / (2.0 * N)

    def fit(self, X, y):
        """Treinamento do ADALINE: adiciona bias; para cada amostra atualiza os pesos;
           interrompe se o ΔEQM ≤ ε ou ao atingir max_epochs.
        """
        N = X.shape[0]
        X_bias = np.hstack((np.ones((N, 1)), X))  # adiciona bias

        self.weights = np.zeros(X_bias.shape[1])
        epoch = 0
        eqm_prev = self._eqm(X_bias, y)
        while epoch < self.max_epochs:
            for i in range(N):
                u = np.dot(self.weights, X_bias[i])
                self.weights += self.eta * (y[i] - u) * X_bias[i]
            epoch += 1
            eqm_current = self._eqm(X_bias, y)
            if np.abs(eqm_current - eqm_prev) <= self.epsilon:
                break
            eqm_prev = eqm_current
        return self

    def predict(self, X):
        """Fase de teste: retorna u = wᵀx (com bias) para cada amostra.
        """
        N = X.shape[0]
        X_bias = np.hstack((np.ones((N, 1)), X))
        return np.dot(X_bias, self.weights)

# ===============================
# 2. IMPLEMENTAÇÃO DO MLP (DO ZERO)
# ===============================
class SimpleMLP:
    def __init__(self, input_dim, hidden_layers, output_dim=1,
                 eta=0.05, max_epochs=200, epsilon=1e-4,
                 activation_name="tanh", random_state=None):
        """
        Configuração do MLP:
         - input_dim: dimensão da entrada (p).
         - hidden_layers: lista com o número de neurônios em cada camada oculta.
         - output_dim: neurônios na camada de saída.
         - Taxa de aprendizagem η = 0.05, max_epochs = 200, ε = 1e-4.
         - activation_name: "tanh" ou "sigmoid" (aqui usamos "tanh").
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.eta = eta
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.activation_name = activation_name
        self.random_state = random_state

        self.L_total = len(hidden_layers) + 1  
        self.weights = []
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicializa as matrizes de peso com valores no intervalo [-0.5, 0.5], 
           incluindo os bias, conforme as dimensões definidas.
        """
        rng = np.random.RandomState(self.random_state)
        # Primeira camada (entrada + bias)
        W0 = rng.uniform(-0.5, 0.5, (self.hidden_layers[0], self.input_dim + 1))
        self.weights.append(W0)
        # Camadas ocultas (se houver mais que uma)
        for i in range(1, len(self.hidden_layers)):
            Wi = rng.uniform(-0.5, 0.5, (self.hidden_layers[i], self.hidden_layers[i-1] + 1))
            self.weights.append(Wi)
        # Camada de saída
        if len(self.hidden_layers) > 0:
            input_to_output = self.hidden_layers[-1] + 1
        else:
            input_to_output = self.input_dim + 1
        W_out = rng.uniform(-0.5, 0.5, (self.output_dim, input_to_output))
        self.weights.append(W_out)

    def activation(self, z, layer):
        """Função de ativação para todos os neurônios:
           - "tanh": np.tanh(z)
           - "sigmoid": 1 / (1+exp(-z))
        """
        if self.activation_name == "tanh":
            return np.tanh(z)
        elif self.activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        else:
            return z

    def activation_derivative(self, z, layer):
        """Derivada da função de ativação:
           - "tanh": 1 - tanh(z)²
           - "sigmoid": sigma(z) * (1 - sigma(z))
        """
        if self.activation_name == "tanh":
            return 1.0 - np.tanh(z)**2
        elif self.activation_name == "sigmoid":
            sig = 1.0 / (1.0 + np.exp(-z))
            return sig * (1 - sig)
        else:
            return np.ones_like(z)

    def forward(self, x):
        """Propagação forward: insere bias (-1) na entrada e propaga por todas as camadas,
           salvando os valores pré-ativação (i_list) e as ativações (y_list).
        """
        self.i_list = []
        self.y_list = []

        x_bias = np.concatenate(([-1], x))
        i0 = np.dot(self.weights[0], x_bias)
        self.i_list.append(i0)
        y0 = self.activation(i0, 0)
        self.y_list.append(y0)

        for j in range(1, len(self.weights)):
            y_prev = self.y_list[j-1]
            y_bias = np.concatenate(([-1], y_prev))
            i_j = np.dot(self.weights[j], y_bias)
            self.i_list.append(i_j)
            y_j = self.activation(i_j, j)
            self.y_list.append(y_j)
        return self.y_list[-1]

    def backward(self, x, d):
        """Backpropagation: calcula os deltas e atualiza os pesos em cada camada.
        """
        L = len(self.weights) - 1
        delta = [None] * (L + 1)

        deriv = self.activation_derivative(self.i_list[L], L)
        delta[L] = deriv * (d - self.y_list[L])
        
        for j in range(L-1, -1, -1):
            W_next = self.weights[j+1]
            Wb = W_next[:, 1:]  # descarta o bias
            deriv_current = self.activation_derivative(self.i_list[j], j)
            delta[j] = deriv_current * np.dot(Wb.T, delta[j+1])
        
        x_bias = np.concatenate(([-1], x))
        self.weights[0] += self.eta * np.outer(delta[0], x_bias)
        for j in range(1, len(self.weights)):
            y_prev = self.y_list[j-1]
            y_bias = np.concatenate(([-1], y_prev))
            self.weights[j] += self.eta * np.outer(delta[j], y_bias)

    def train(self, X, Y):
        """Treinamento do MLP: para cada época e para cada amostra, realiza forward, 
           backpropagation e acumula o EQM (loss_curve). Interrompe se EQM < ε ou atinge max_epochs.
        """
        N = X.shape[0]
        epoch = 0
        EQM = 1.0
        self.loss_curve = []
        while EQM > self.epsilon and epoch < self.max_epochs:
            soma_erros = 0.0
            for i in range(N):
                x_sample = X[i]
                d_sample = Y[i]
                output = self.forward(x_sample)
                erro_sample = d_sample - output
                soma_erros += np.sum(erro_sample**2)
                self.backward(x_sample, d_sample)
            EQM = soma_erros / (2.0 * N)
            self.loss_curve.append(EQM)
            epoch += 1
        return self

    def predict(self, X):
        """Para cada amostra em X, realiza forward e retorna a previsão.
        """
        outputs = []
        for i in range(X.shape[0]):
            x_sample = X[i]
            outputs.append(self.forward(x_sample))
        return np.array(outputs)

# ===============================
# 3. FUNÇÕES AUXILIARES
# ===============================
def load_data(filepath):
    """Lê o arquivo de dados e organiza as variáveis:
           X: velocidade do vento (vetor coluna)
           y: potência gerada (vetor)
    """
    data = np.loadtxt(filepath)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    return X, y

def initial_scatter_plot(X, y):
    """Exibe um gráfico de dispersão dos dados normalizados.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', edgecolor='k', alpha=0.7)
    plt.xlabel("Velocidade do Vento (normalizada)")
    plt.ylabel("Potência Gerada (normalizada)")
    plt.title("Relação: Velocidade do Vento vs Potência Gerada")
    plt.show()

def compute_mse(y_true, y_pred):
    """Calcula o MSE.
    """
    return np.mean((y_true - y_pred) ** 2)

def monte_carlo_simulation(X, y, R=250):
    """
    Executa R simulações Monte Carlo com divisão 80%/20%,
    treina os modelos ADALINE e MLP e retorna as listas dos MSEs.
    Exibe também o progresso no console.
    """
    n_samples = X.shape[0]
    mse_adaline_list = []
    mse_mlp_list = []
    
    for r in range(R):
        indices = np.random.permutation(n_samples)
        split = int(0.8 * n_samples)
        train_idx = indices[:split]
        test_idx = indices[split:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # ADALINE otimizado
        adaline = Adaline(eta=0.01, max_epochs=100, epsilon=1e-4, random_state=r)
        adaline.fit(X_train, y_train)
        y_pred_adaline = adaline.predict(X_test)
        mse_adaline = compute_mse(y_test, y_pred_adaline)
        mse_adaline_list.append(mse_adaline)
        
        # MLP otimizado
        mlp = SimpleMLP(input_dim=1, hidden_layers=[10], output_dim=1,
                        eta=0.05, max_epochs=200, epsilon=1e-4,
                        activation_name="tanh", random_state=r)
        mlp.train(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test).squeeze()
        mse_mlp = compute_mse(y_test, y_pred_mlp)
        mse_mlp_list.append(mse_mlp)
        
        progress = ((r + 1) / R) * 100.0
        sys.stdout.write(f"\rProgresso Monte Carlo: {progress:5.1f}%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return mse_adaline_list, mse_mlp_list

def print_statistics(mse_list):
    """Retorna média, desvio-padrão, maior e menor valor dos MSEs.
    """
    mean_val = np.mean(mse_list)
    std_val = np.std(mse_list)
    max_val = np.max(mse_list)
    min_val = np.min(mse_list)
    return mean_val, std_val, max_val, min_val

def plot_learning_curve(curve_data, labels, title):
    """Plota uma ou mais curvas de aprendizado.
    curve_data: lista de listas dos valores de EQM por época (uma por modelo).
    labels: lista de strings para a legenda.
    """
    plt.figure(figsize=(10, 6))
    for data, label in zip(curve_data, labels):
        plt.plot(data, marker='o', linestyle='-', label=label)
    plt.xlabel("Épocas")
    plt.ylabel("EQM")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# ===============================
# 4. FUNÇÃO PRINCIPAL (MAIN)
# ===============================
def main():
    filepath = r"dados/aerogerador.dat"
    X, y = load_data(filepath)
    
    # Normalização dos dados
    X_norm, y_norm = normalize_data(X, y)
    
    # Exibe o gráfico de dispersão
    initial_scatter_plot(X_norm, y_norm)
    
    # Executa a simulação Monte Carlo (R = 250)
    R = 250
    mse_adaline_list, mse_mlp_list = monte_carlo_simulation(X_norm, y_norm, R=R)
    
    adaline_stats = print_statistics(mse_adaline_list)
    mlp_stats = print_statistics(mse_mlp_list)
    
    print("{:<40} {:>10} {:>15} {:>15} {:>15}".format("Modelo", "Média", "Desvio-Padrão", "Maior Valor", "Menor Valor"))
    print("-" * 90)
    print("{:<40} {:10.4f} {:15.4f} {:15.4f} {:15.4f}".format("ADALINE",
                                                              adaline_stats[0],
                                                              adaline_stats[1],
                                                              adaline_stats[2],
                                                              adaline_stats[3]))
    print("{:<40} {:10.4f} {:15.4f} {:15.4f} {:15.4f}".format("MLP (Perceptron Multicamadas)",
                                                              mlp_stats[0],
                                                              mlp_stats[1],
                                                              mlp_stats[2],
                                                              mlp_stats[3]))
    
    # ----------------------------------------------------------------
    # Demonstração de Underfitting e Overfitting na MLP
    # ----------------------------------------------------------------
    # Para análise, vamos definir duas topologias:
    # (a) MLP Subdimensionado: 1 camada oculta com 1 neurônio – tendência a underfitting.
    # (b) MLP Superdimensionado: 2 camadas ocultas com 50 neurônios cada – tendência a overfitting.
    
    # Obtemos uma divisão fixa (random_state=42)
    np.random.seed(42)
    indices = np.random.permutation(X_norm.shape[0])
    split = int(0.8 * X_norm.shape[0])
    train_idx = indices[:split]
    X_train, y_train = X_norm[train_idx], y_norm[train_idx]
    
    # MLP subdimensionado
    mlp_sub = SimpleMLP(input_dim=1, hidden_layers=[1], output_dim=1,
                        eta=0.05, max_epochs=200, epsilon=1e-4,
                        activation_name="tanh", random_state=42)
    mlp_sub.train(X_train, y_train)
    
    # MLP superdimensionado
    mlp_super = SimpleMLP(input_dim=1, hidden_layers=[50, 50], output_dim=1,
                          eta=0.05, max_epochs=200, epsilon=1e-4,
                          activation_name="tanh", random_state=42)
    mlp_super.train(X_train, y_train)
    
    # Plot das curvas de aprendizado
    curve_data = [mlp_sub.loss_curve, mlp_super.loss_curve]
    labels = ["MLP Subdimensionado (1 neurônio)", "MLP Superdimensionado (50,50 neurônios)"]
    title = "Curvas de Aprendizado da MLP: Underfitting vs Overfitting"
    plot_learning_curve(curve_data, labels, title)
    
if __name__ == '__main__':
    main()
