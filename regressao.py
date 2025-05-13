#!/usr/bin/env python
# regressao.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# ===============================
# Implementação do ADALINE
# ===============================
class Adaline:
    def __init__(self, eta=0.001, max_epochs=100, epsilon=1e-5, random_state=1):
        """
        Parâmetros:
          eta         : Taxa de aprendizagem.
          max_epochs  : Número máximo de épocas.
          epsilon     : Critério de convergência (diferença entre EQMs consecutivos).
          random_state: Semente para a inicialização (quando aplicável).
        """
        self.eta = eta
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.random_state = random_state
        self.weights = None  # vetor de pesos (incluirá o bias)

    def _eqm(self, X, y):
        """
        Calcula o Erro Quadrático Médio (EQM) conforme o Algoritmo 4.
        Para cada amostra, calcula u = w^T * x e acumula (d - u)².
        Ao final, divide a soma por (2*N).
        """
        N = X.shape[0]
        eqm = 0.0
        for i in range(N):
            u = np.dot(self.weights, X[i])
            eqm += (y[i] - u) ** 2
        return eqm / (2.0 * N)

    def fit(self, X, y):
        """
        Treinamento do ADALINE conforme Algoritmo 3:
          - Adiciona a coluna de bias à matriz X.
          - Inicializa os pesos (aqui, com zeros).
          - Para cada época, percorre todas as amostras e atualiza os pesos.
          - O treinamento para quando a variação do EQM é menor que epsilon ou
            atinge o número máximo de épocas.
        """
        N = X.shape[0]
        # Adiciona coluna de bias (utiliza 1 para facilitar o cálculo)
        X_bias = np.hstack((np.ones((N, 1)), X))
        
        # Inicializa os pesos
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
        """
        Fase de teste (Algoritmo 5):
          - Adiciona a coluna de bias e retorna u = w^T * x para cada amostra.
          - Em classificação, poder-se-ia aplicar uma função de sinal; aqui, retorna
            o valor contínuo para regressão.
        """
        N = X.shape[0]
        X_bias = np.hstack((np.ones((N, 1)), X))
        return np.dot(X_bias, self.weights)

# ===============================
# Implementação da MLP (do zero)
# ===============================
class SimpleMLP:
    def __init__(self, input_dim, hidden_layers, output_dim=1,
                 eta=0.01, max_epochs=1000, epsilon=1e-5, random_state=None):
        """
        Parâmetros:
          input_dim    : Dimensão da entrada (p)
          hidden_layers: Lista com a quantidade de neurônios em cada camada oculta (ex.: [10])
          output_dim   : Quantidade de neurônios na camada de saída (para regressão, geralmente 1)
          eta          : Taxa de aprendizagem.
          max_epochs   : Número máximo de épocas.
          epsilon      : Critério de parada com base no EQM.
          random_state : Semente para reprodutibilidade.
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.eta = eta
        self.max_epochs = max_epochs
        self.epsilon = epsilon
        self.random_state = random_state

        # Total de camadas (camadas ocultas + camada de saída)
        self.L_total = len(hidden_layers) + 1  
        self.weights = []  # Lista de matrizes de pesos para cada camada

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Conforme o Algoritmo 6, cria L+1 matrizes de pesos, inicializadas com valores
        aleatórios pequenos no intervalo [-0.5, 0.5]. Cada matriz inclui os pesos do bias.
        """
        rng = np.random.RandomState(self.random_state)
        # Primeira camada: entrada + bias
        W0 = rng.uniform(-0.5, 0.5, (self.hidden_layers[0], self.input_dim + 1))
        self.weights.append(W0)
        # Camadas ocultas intermediárias (se houver)
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
        """
        Função de ativação:
          - Para camadas ocultas (layer < último índice): tanh.
          - Para a camada de saída: ativação linear (para regressão).
        """
        if layer < len(self.weights) - 1:
            return np.tanh(z)
        else:
            return z

    def activation_derivative(self, z, layer):
        """
        Derivada da função de ativação:
          - Para tanh: 1 - tanh(z)^2.
          - Para a camada de saída (linear): 1.
        """
        if layer < len(self.weights) - 1:
            return 1.0 - np.tanh(z)**2
        else:
            return np.ones_like(z)

    def forward(self, x):
        """
        Propagação forward conforme o Algoritmo 8:
          - Adiciona o bias (–1) à amostra.
          - Propaga a entrada por todas as camadas, salvando os valores pré-ativação (i)
            e as ativações (y).
        """
        self.i_list = []  # lista dos valores pré-ativação
        self.y_list = []  # lista dos valores de ativação

        # Primeira camada
        x_bias = np.concatenate(([-1], x))
        i0 = np.dot(self.weights[0], x_bias)
        self.i_list.append(i0)
        y0 = self.activation(i0, 0)
        self.y_list.append(y0)

        # Camadas seguintes
        for j in range(1, len(self.weights)):
            y_prev = self.y_list[j-1]
            # Adiciona o bias à saída da camada anterior
            y_bias = np.concatenate(([-1], y_prev))
            i_j = np.dot(self.weights[j], y_bias)
            self.i_list.append(i_j)
            y_j = self.activation(i_j, j)
            self.y_list.append(y_j)
        return self.y_list[-1]

    def backward(self, x, d):
        """
        Backpropagation conforme o Algoritmo 9:
          - Calcula os deltas (δ) para cada camada (da camada de saída para as ocultas)
          - Atualiza os pesos usando a regra: W[j] ← W[j] + η * (δ[j] ⊗ (entrada da camada)).
        """
        L = len(self.weights) - 1  # índice da camada de saída
        delta = [None] * (L + 1)
        # Camada de saída:
        deriv = self.activation_derivative(self.i_list[L], L)
        delta[L] = deriv * (d - self.y_list[L])
        
        # Camadas ocultas (retropropagação)
        for j in range(L-1, -1, -1):
            # Remove o peso do bias da camada seguinte
            W_next = self.weights[j+1]
            Wb = W_next[:, 1:]
            deriv_current = self.activation_derivative(self.i_list[j], j)
            delta[j] = deriv_current * np.dot(Wb.T, delta[j+1])
        
        # Atualização dos pesos
        # Para a camada 0:
        x_bias = np.concatenate(([-1], x))
        self.weights[0] += self.eta * np.outer(delta[0], x_bias)
        # Para as demais camadas:
        for j in range(1, len(self.weights)):
            y_prev = self.y_list[j-1]
            y_bias = np.concatenate(([-1], y_prev))
            self.weights[j] += self.eta * np.outer(delta[j], y_bias)

    def train(self, X, Y):
        """
        Treinamento da rede MLP conforme os Algoritmos 7 e 10:
          - Em cada época, para cada amostra, realiza a propagação forward, calcula o erro
            e aplica o backpropagation.
          - O EQM é calculado por época; o treinamento para quando o EQM for inferior a epsilon ou
            quando o número máximo de épocas for atingido.
          - A curva de aprendizado (EQM por época) é salva em self.loss_curve.
        """
        N = X.shape[0]
        epoch = 0
        EQM = 1.0  # valor inicial arbitrário
        self.loss_curve = []
        while EQM > self.epsilon and epoch < self.max_epochs:
            soma_erros = 0.0
            for i in range(N):
                x_sample = X[i]         # amostra com dimensão (input_dim,)
                d_sample = Y[i]         # saída desejada (para regressão)
                output = self.forward(x_sample)
                erro_sample = d_sample - output
                soma_erros += np.sum(erro_sample**2)
                self.backward(x_sample, d_sample)
            EQM = soma_erros / (2.0 * N)
            self.loss_curve.append(EQM)
            epoch += 1
        return self

    def predict(self, X):
        """
        Fase de teste (Algoritmo 11):
          - Para cada amostra, realiza a propagação forward e retorna o valor de saída.
        """
        outputs = []
        for i in range(X.shape[0]):
            x_sample = X[i]
            outputs.append(self.forward(x_sample))
        return np.array(outputs)

# ===============================
# Funções de apoio
# ===============================
def load_data(filepath):
    """
    Lê o arquivo de dados e organiza as variáveis:
      - X: Velocidade do vento (vetor coluna).
      - y: Potência gerada (vetor).
    """
    data = np.loadtxt(filepath)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    return X, y

def initial_scatter_plot(X, y):
    """
    Exibe um gráfico de dispersão usando seaborn/matplotlib.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', edgecolor='k', alpha=0.7)
    plt.xlabel("Velocidade do Vento")
    plt.ylabel("Potência Gerada")
    plt.title("Relação: Velocidade do Vento vs Potência Gerada")
    plt.show()

def compute_mse(y_true, y_pred):
    """
    Calcula o Erro Quadrático Médio (MSE).
    """
    return np.mean((y_true - y_pred) ** 2)

def monte_carlo_simulation(X, y, R=250):
    """
    Realiza R simulações Monte Carlo:
      - Em cada iteração, divide os dados em 80% treino e 20% teste,
        treina os modelos ADALINE e MLP e computa o MSE.
      - Exibe o progresso (%) em cada iteração.
    Retorna duas listas de MSE para cada modelo.
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
        
        # Treinamento do ADALINE
        adaline = Adaline(eta=0.0001, max_epochs=200, epsilon=1e-5, random_state=r)
        adaline.fit(X_train, y_train)
        y_pred_adaline = adaline.predict(X_test)
        mse_adaline = compute_mse(y_test, y_pred_adaline)
        mse_adaline_list.append(mse_adaline)
        
        # Treinamento do MLP com topologia: 1 camada oculta com 10 neurônios
        mlp = SimpleMLP(input_dim=1, hidden_layers=[10], output_dim=1,
                        eta=0.01, max_epochs=300, epsilon=1e-5, random_state=r)
        mlp.train(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test)
        y_pred_mlp = np.array(y_pred_mlp).squeeze()  # Ajusta a forma para 1D
        mse_mlp = compute_mse(y_test, y_pred_mlp)
        mse_mlp_list.append(mse_mlp)
        
        # Exibe o progresso (atualiza a mesma linha)
        progress = ((r + 1) / R) * 100.0
        sys.stdout.write(f"\rProgresso Monte Carlo: {progress:5.1f}%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return mse_adaline_list, mse_mlp_list

def print_statistics(mse_list):
    """
    Retorna a média, desvio-padrão, maior e menor valor dos MSEs.
    """
    mean_val = np.mean(mse_list)
    std_val = np.std(mse_list)
    max_val = np.max(mse_list)
    min_val = np.min(mse_list)
    return mean_val, std_val, max_val, min_val

def plot_learning_curve(mlp):
    """
    Exibe a curva de aprendizado (EQM vs épocas) do MLP.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(mlp.loss_curve, marker='o', linestyle='-', color='b')
    plt.xlabel("Épocas")
    plt.ylabel("EQM (Erro Quadrático Médio)")
    plt.title("Curva de Aprendizado do MLP")
    plt.grid(True)
    plt.show()

# ===============================
# Função principal (main)
# ===============================
def main():
    # Caminho do arquivo de dados
    filepath = r"C:\Users\Bruno Matos\iCloudDrive\UNIFOR\SEMESTRE 6\Inteligência artificial computacional\AV2\Redes_Neurais_Artificiais\dados\aerogerador.dat"
    
    # Carregar os dados
    X, y = load_data(filepath)
    
    # Exibe o gráfico de dispersão inicial
    initial_scatter_plot(X, y)
    
    # Realiza a validação Monte Carlo (R = 250)
    R = 250
    mse_adaline_list, mse_mlp_list = monte_carlo_simulation(X, y, R=R)
    
    # Calcula estatísticas dos MSEs para cada modelo
    adaline_stats = print_statistics(mse_adaline_list)
    mlp_stats = print_statistics(mse_mlp_list)
    
    # Imprime os resultados em formato tabular no console
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
    
    # Exemplo: Treinamento fixo do MLP para exibir a curva de aprendizado
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    split = int(0.8 * X.shape[0])
    train_idx = indices[:split]
    X_train, y_train = X[train_idx], y[train_idx]
    
    mlp_example = SimpleMLP(input_dim=1, hidden_layers=[10], output_dim=1,
                            eta=0.01, max_epochs=300, epsilon=1e-5, random_state=42)
    mlp_example.train(X_train, y_train)
    plot_learning_curve(mlp_example)

if __name__ == '__main__':
    main()
