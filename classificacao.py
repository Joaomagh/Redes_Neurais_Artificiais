import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from mpl_toolkits.mplot3d import Axes3D  # para plotagem 3D

# -------------------------------
# FUNÇÕES AUXILIARES PARA DADOS E MÉTRICAS
# -------------------------------
def load_classification_data(filepath):
    """
    Lê o arquivo "Spiral3d.csv" e organiza:
      - X: colunas 1 a 3 (features)
      - y: coluna 4 (rótulo)
    """
    data = np.loadtxt(filepath, delimiter=",")
    X = data[:, :3]
    y = data[:, 3]
    return X, y

def convert_labels(y):
    """
    Se os rótulos forem 0 e 1, converte para -1 e 1, respectivamente.
    Caso contrário, assume que já estão em {-1,1}.
    """
    unique = np.unique(y)
    if set(unique) == set([0, 1]):
        return np.where(y == 0, -1, 1)
    return y

def plot_3d_scatter(X, y):
    """
    Plota um gráfico de dispersão 3D dos dados com cores de acordo com a classe.
    """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    # Converter y para string para colorir conforme classe
    scatter = ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.set_title("Gráfico 3D de Dispersão")
    plt.colorbar(scatter, label="Classe")
    plt.show()

def compute_confusion_matrix(y_true, y_pred):
    """
    Calcula a matriz de confusão para duas classes (assumindo rótulos -1 e 1)
    Retorna uma matriz 2x2:
         [[TN, FP],
          [FN, TP]]
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    return np.array([[TN, FP], [FN, TP]])

def compute_classification_metrics(y_true, y_pred):
    """
    Calcula: acurácia, sensibilidade (recall para classe positiva) e especificidade.
    Assume rótulos -1 (negativo) e 1 (positivo).
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    return accuracy, sensitivity, specificity, cm

# -------------------------------
# CLASSE PERCEPTRON SIMPLES (CLASSIFICAÇÃO)
# -------------------------------
class PerceptronSimple:
    def __init__(self, eta=0.1, max_epochs=100, random_state=None):
        """
        Inicializa o Perceptron Simples.
        
        Parâmetros:
          - eta: taxa de aprendizagem (valor entre 0 e 1)
          - max_epochs: número máximo de épocas
          - random_state: semente para reprodução (opcional)
          
        A implementação segue o pseudocódigo:
          1. Inicializa os pesos (com bias)
          2. Enquanto houver algum erro em uma época, atualize os pesos
        """
        self.eta = eta
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.weights = None      # Vetor de pesos (incluindo o bias)
        self.errors_curve = []   # Armazena o número de atualizações (erros) por época

    def net_input(self, x):
        """Calcula a entrada líquida: u = w^T * x (incluindo o bias)"""
        return np.dot(self.weights[1:], x) + self.weights[0]

    def predict(self, X):
        """Aplica a função sinal em u. Retorna 1 se u >= 0; do contrário, -1."""
        # Para uma matriz de amostras:
        return np.where(np.dot(X, self.weights[1:]) + self.weights[0] >= 0, 1, -1)

    def fit(self, X, y):
        """
        Treinamento do Perceptron Simples conforme o Algorithm 1:
          1. Inicialize os pesos (com bias) com zeros ou valores aleatórios.
          2. Defina a flag ERRO como 'EXISTENTE'.
          3. Enquanto existir erro:
               a. Para cada amostra, calcule:
                  u = w^T * x
                  y_pred = signal(u)
               b. Se y_pred for diferente de d:
                  Atualize: w = w + η (d - y_pred)x
                  Marque que erro existe.
               c. Incrementa o contador de épocas.
        O laço termina quando uma época inteira não gera nenhuma atualização.
        """
        n_samples, n_features = X.shape
        # Inicializa os pesos. Usamos zeros aqui; também poderia ser aleatório.
        self.weights = np.zeros(n_features + 1)
        epoch = 0
        
        # Inicia com o critério ERRO "EXISTENTE"
        error_exists = True

        while error_exists and epoch < self.max_epochs:
            error_exists = False
            errors_epoch = 0  # Contador de erros na época corrente
            for xi, target in zip(X, y):
                u = self.net_input(xi)
                # Função sinal: se u >= 0, então 1; caso contrário, -1.
                prediction = 1 if u >= 0 else -1
                # Se a previsão estiver errada, atualiza os pesos
                if target != prediction:
                    update = self.eta * (target - prediction)
                    self.weights[1:] += update * xi
                    self.weights[0] += update
                    errors_epoch += 1
                    error_exists = True
            self.errors_curve.append(errors_epoch)
            epoch += 1
        return self


# -------------------------------
# CLASSE MLP SIMPLES (CLASSIFICAÇÃO)
# -------------------------------
class SimpleMLP:
    def __init__(self, input_dim, hidden_layers, output_dim=1,
                 eta=0.1, max_epochs=200, epsilon=1e-4, activation_name="tanh",
                 random_state=None):
        """
        MLP para classificação:
         - input_dim: dimensão dos dados de entrada.
         - hidden_layers: lista com número de neurônios em cada camada oculta.
         - output_dim: normalmente 1 para classificação binária.
         - eta, max_epochs, epsilon: parâmetros de treinamento.
         - activation_name: "tanh" (pode escolher também "sigmoid").
         - Armazena a curva de aprendizado (loss_curve).
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
        self.loss_curve = []
        
    def _initialize_weights(self):
        rgen = np.random.RandomState(self.random_state)
        W0 = rgen.uniform(-0.5, 0.5, (self.hidden_layers[0], self.input_dim + 1))
        self.weights.append(W0)
        for i in range(1, len(self.hidden_layers)):
            Wi = rgen.uniform(-0.5, 0.5, (self.hidden_layers[i], self.hidden_layers[i-1] + 1))
            self.weights.append(Wi)
        if len(self.hidden_layers) > 0:
            inp_out = self.hidden_layers[-1] + 1
        else:
            inp_out = self.input_dim + 1
        W_out = rgen.uniform(-0.5, 0.5, (self.output_dim, inp_out))
        self.weights.append(W_out)
        
    def activation(self, z):
        if self.activation_name == "tanh":
            return np.tanh(z)
        elif self.activation_name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        else:
            return z  # linear
    
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
        
        # Primeira camada: adicione bias -1
        x_bias = np.concatenate(([-1], x))
        i0 = np.dot(self.weights[0], x_bias)
        self.i_list.append(i0)
        y0 = self.activation(i0)
        self.y_list.append(y0)
        
        for j in range(1, len(self.weights)):
            y_prev = self.y_list[j-1]
            y_bias = np.concatenate(([-1], y_prev))
            i_j = np.dot(self.weights[j], y_bias)
            self.i_list.append(i_j)
            y_j = self.activation(i_j)
            self.y_list.append(y_j)
        return self.y_list[-1]
    
    def backward(self, x, d):
        L = len(self.weights) - 1
        delta = [None] * (L + 1)
        
        # Cálculo para a camada de saída
        delta[L] = self.activation_derivative(self.i_list[L]) * (d - self.y_list[L])
        for j in range(L-1, -1, -1):
            W_next = self.weights[j+1]
            W_no_bias = W_next[:, 1:]
            delta[j] = self.activation_derivative(self.i_list[j]) * np.dot(W_no_bias.T, delta[j+1])
            
        x_bias = np.concatenate(([-1], x))
        self.weights[0] += self.eta * np.outer(delta[0], x_bias)
        for j in range(1, len(self.weights)):
            y_prev = self.y_list[j-1]
            y_bias = np.concatenate(([-1], y_prev))
            self.weights[j] += self.eta * np.outer(delta[j], y_bias)
    
    def train(self, X, Y):
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
                erro = d_sample - output
                soma_erros += np.sum(erro**2)
                self.backward(x_sample, d_sample)
            EQM = soma_erros / (2.0 * N)
            self.loss_curve.append(EQM)
            epoch += 1
        return self

    def predict(self, X):
        outputs = []
        for i in range(X.shape[0]):
            x_sample = X[i]
            outputs.append(self.forward(x_sample))
        # Saída contínua; para classificação, use threshold=0:
        outputs = np.array(outputs).squeeze()
        preds = np.where(outputs >= 0, 1, -1)
        return preds

# -------------------------------
# FUNÇÃO PARA MONTE CARLO NA CLASSIFICAÇÃO
# -------------------------------
def monte_carlo_classification(X, y, R=250):
    """
    Executa R rodadas de validação Monte Carlo:
      - Em cada rodada, embaralha e divide os dados (80% treino, 20% teste).
      - Treina Perceptron Simples e MLP (topologia padrão: 1 camada oculta com 10 neurônios).
      - Computa acurácia, sensibilidade e especificidade.
      - Armazena também os índices da rodada com maior e menor acurácia para cada método, assim como
        as respectivas curvas de aprendizado e matrizes de confusão.
    """
    n_samples = X.shape[0]
    
    # listas para métricas (cada item: (accuracy, sensitivity, specificity)).
    metrics_percep = []
    metrics_mlp = []
    
    # para guardar as curvas de aprendizado de melhor e pior rodada
    best_percep = {"acc": -1, "index": None, "cm": None, "curve": None, "y_test": None, "y_pred": None}
    worst_percep = {"acc": 2, "index": None, "cm": None, "curve": None, "y_test": None, "y_pred": None}
    
    best_mlp = {"acc": -1, "index": None, "cm": None, "curve": None, "y_test": None, "y_pred": None}
    worst_mlp = {"acc": 2, "index": None, "cm": None, "curve": None, "y_test": None, "y_pred": None}
    
    for r in range(R):
        indices = np.random.permutation(n_samples)
        split = int(0.8 * n_samples)
        train_idx = indices[:split]
        test_idx = indices[split:]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Perceptron Simples
        percep = PerceptronSimple(eta=0.1, max_epochs=100, random_state=r)
        percep.fit(X_train, y_train)
        y_pred_percep = percep.predict(X_test)
        acc, sens, spec, cm = compute_classification_metrics(y_test, y_pred_percep)
        metrics_percep.append((acc, sens, spec))
        
        if acc > best_percep["acc"]:
            best_percep["acc"] = acc
            best_percep["index"] = r
            best_percep["cm"] = cm
            best_percep["curve"] = percep.errors_curve.copy()
            best_percep["y_test"] = y_test.copy()
            best_percep["y_pred"] = y_pred_percep.copy()
        if acc < worst_percep["acc"]:
            worst_percep["acc"] = acc
            worst_percep["index"] = r
            worst_percep["cm"] = cm
            worst_percep["curve"] = percep.errors_curve.copy()
            worst_percep["y_test"] = y_test.copy()
            worst_percep["y_pred"] = y_pred_percep.copy()
        
        # MLP com topologia padrão: 1 camada oculta com 10 neurônios
        mlp = SimpleMLP(input_dim=3, hidden_layers=[10], output_dim=1,
                        eta=0.1, max_epochs=200, epsilon=1e-4,
                        activation_name="tanh", random_state=r)
        mlp.train(X_train, y_train)
        y_pred_mlp = mlp.predict(X_test)
        acc_mlp, sens_mlp, spec_mlp, cm_mlp = compute_classification_metrics(y_test, y_pred_mlp)
        metrics_mlp.append((acc_mlp, sens_mlp, spec_mlp))
        
        if acc_mlp > best_mlp["acc"]:
            best_mlp["acc"] = acc_mlp
            best_mlp["index"] = r
            best_mlp["cm"] = cm_mlp
            best_mlp["curve"] = mlp.loss_curve.copy()
            best_mlp["y_test"] = y_test.copy()
            best_mlp["y_pred"] = y_pred_mlp.copy()
        if acc_mlp < worst_mlp["acc"]:
            worst_mlp["acc"] = acc_mlp
            worst_mlp["index"] = r
            worst_mlp["cm"] = cm_mlp
            worst_mlp["curve"] = mlp.loss_curve.copy()
            worst_mlp["y_test"] = y_test.copy()
            worst_mlp["y_pred"] = y_pred_mlp.copy()
            
        progress = ((r+1)/R)*100.0
        sys.stdout.write(f"\rMonte Carlo Progress: {progress:5.1f}%")
        sys.stdout.flush()
    sys.stdout.write("\n")
    return metrics_percep, metrics_mlp, best_percep, worst_percep, best_mlp, worst_mlp

# -------------------------------
# FUNÇÃO PARA PLOTAR CONTOUR DA MATRIZ DE CONFUSÃO
# -------------------------------
def plot_confusion_matrix(cm, title="Matriz de Confusão"):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred -1", "Pred 1"],
                yticklabels=["Real -1", "Real 1"])
    plt.title(title)
    plt.ylabel("Valor Real")
    plt.xlabel("Valor Predito")
    plt.show()

# -------------------------------
# FUNÇÃO PARA PLOTAR CURVA DE APRENDIZADO
# -------------------------------
def plot_learning_curve_single(curve, title="Curva de Aprendizado"):
    plt.figure(figsize=(8,6))
    plt.plot(curve, marker='o', linestyle='-', color='b')
    plt.xlabel("Épocas")
    plt.ylabel("Erro ou Loss")
    plt.title(title)
    plt.grid(True)
    plt.show()

# -------------------------------
# FUNÇÃO PARA EXPERIMENTOS DE TOPOLOGIAS (UNDERFITTING e OVERFITTING)
# -------------------------------
def experiment_topologies(X_train, y_train, X_test, y_test):
    """
    Treina duas MLP com topologias diferentes:
      (a) MLP Subdimensionada (underfitting): 1 camada oculta com 1 neurônio.
      (b) MLP Superdimensionada (overfitting): 2 camadas ocultas com 50 neurônios cada.
    Calcula as métricas e plota as curvas de aprendizado e a matriz de confusão.
    """
    # MLP subdimensionada
    mlp_under = SimpleMLP(input_dim=3, hidden_layers=[1], output_dim=1,
                           eta=0.1, max_epochs=200, epsilon=1e-4,
                           activation_name="tanh", random_state=42)
    mlp_under.train(X_train, y_train)
    y_pred_under = mlp_under.predict(X_test)
    acc_under, sens_under, spec_under, cm_under = compute_classification_metrics(y_test, y_pred_under)
    
    # MLP superdimensionada
    mlp_over = SimpleMLP(input_dim=3, hidden_layers=[50, 50], output_dim=1,
                          eta=0.1, max_epochs=200, epsilon=1e-4,
                          activation_name="tanh", random_state=42)
    mlp_over.train(X_train, y_train)
    y_pred_over = mlp_over.predict(X_test)
    acc_over, sens_over, spec_over, cm_over = compute_classification_metrics(y_test, y_pred_over)
    
    print("Topologia Underfitting (MLP Subdimensionado):")
    print(f"Acurácia: {acc_under:.4f}, Sensibilidade: {sens_under:.4f}, Especificidade: {spec_under:.4f}")
    plot_confusion_matrix(cm_under, title="Matriz de Confusão - MLP Subdimensionada")
    plot_learning_curve_single(mlp_under.loss_curve, title="Curva de Aprendizado - MLP Subdimensionada")
    
    print("Topologia Overfitting (MLP Superdimensionado):")
    print(f"Acurácia: {acc_over:.4f}, Sensibilidade: {sens_over:.4f}, Especificidade: {spec_over:.4f}")
    plot_confusion_matrix(cm_over, title="Matriz de Confusão - MLP Superdimensionada")
    plot_learning_curve_single(mlp_over.loss_curve, title="Curva de Aprendizado - MLP Superdimensionada")

# -------------------------------
# FUNÇÃO PRINCIPAL (MAIN)
# -------------------------------
def main():
    # Caminho do arquivo
    filepath = r"C:\Users\Bruno Matos\iCloudDrive\UNIFOR\SEMESTRE 6\Inteligência artificial computacional\AV2\Redes_Neurais_Artificiais\dados\Spiral3d.csv"
    X, y = load_classification_data(filepath)
    y = convert_labels(y)
    
    # Visualização inicial: gráfico 3D de dispersão
    plot_3d_scatter(X, y)
    
    # Monte Carlo: R=250 rodadas
    (metrics_percep, metrics_mlp,
     best_percep, worst_percep,
     best_mlp, worst_mlp) = monte_carlo_classification(X, y, R=250)
    
    # Resumo statístico – separar métricas para cada modelo
    metrics_percep = np.array(metrics_percep)   # colunas: acurácia, sensibilidade, especificidade
    metrics_mlp = np.array(metrics_mlp)
    
    def print_summary(model_name, metrics):
        mean_vals = np.mean(metrics, axis=0)
        std_vals = np.std(metrics, axis=0)
        max_vals = np.max(metrics, axis=0)
        min_vals = np.min(metrics, axis=0)
        print(f"\nResumo das Métricas para {model_name}:")
        print("{:<15} {:>10} {:>15} {:>15} {:>15}".format("Métrica", "Média", "Desvio-Padrão", "Maior Valor", "Menor Valor"))
        print("-"*70)
        labels = ["Acurácia", "Sensibilidade", "Especificidade"]
        for i, lab in enumerate(labels):
            print("{:<15} {:10.4f} {:15.4f} {:15.4f} {:15.4f}".format(lab, mean_vals[i], std_vals[i], max_vals[i], min_vals[i]))
    
    print_summary("Perceptron Simples", metrics_percep)
    print_summary("MLP", metrics_mlp)
    
    # Análise das melhores e piores rodadas:
    print("\n--- Análise das Rodadas (Perceptron Simples) ---")
    print(f"Melhor rodada: {best_percep['index']} com acurácia = {best_percep['acc']:.4f}")
    plot_confusion_matrix(best_percep["cm"], title="Matriz de Confusão - Melhor % (Perceptron)")
    plot_learning_curve_single(best_percep["curve"], title="Curva de Aprendizado - Melhor % (Perceptron)")
    
    print(f"Pior rodada: {worst_percep['index']} com acurácia = {worst_percep['acc']:.4f}")
    plot_confusion_matrix(worst_percep["cm"], title="Matriz de Confusão - Pior % (Perceptron)")
    plot_learning_curve_single(worst_percep["curve"], title="Curva de Aprendizado - Pior % (Perceptron)")
    
    print("\n--- Análise das Rodadas (MLP) ---")
    print(f"Melhor rodada: {best_mlp['index']} com acurácia = {best_mlp['acc']:.4f}")
    plot_confusion_matrix(best_mlp["cm"], title="Matriz de Confusão - Melhor % (MLP)")
    plot_learning_curve_single(best_mlp["curve"], title="Curva de Aprendizado - Melhor % (MLP)")
    
    print(f"Pior rodada: {worst_mlp['index']} com acurácia = {worst_mlp['acc']:.4f}")
    plot_confusion_matrix(worst_mlp["cm"], title="Matriz de Confusão - Pior % (MLP)")
    plot_learning_curve_single(worst_mlp["curve"], title="Curva de Aprendizado - Pior % (MLP)")
    
    # Experimento complementar para MLP: Underfitting vs Overfitting
    # Nesta parte usamos uma divisão fixa para treinar e testar.
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])
    split = int(0.8 * X.shape[0])
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print("\n--- Experimento MLP: Underfitting vs Overfitting ---")
    experiment_topologies(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()
