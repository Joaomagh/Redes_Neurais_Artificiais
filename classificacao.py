import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. Carregar o conjunto de dados
# ==============================
file_path = "Spiral3d.csv"

# Carregar os dados corretamente com delimitador ","
try:
    data = np.loadtxt(file_path, delimiter=",")
    print("Dados carregados com sucesso!")
except Exception as e:
    print("Erro ao carregar o arquivo:", e)
    exit()

# ==============================
# 2. Separar atributos e classes
# ==============================
atributos = data[:, :3]  # As três primeiras colunas são atributos
classes = data[:, 3]     # Última coluna representa as classes (-1 ou 1)

# ==============================
# 3. Visualização dos dados
# ==============================
def plot_dados(atributos, classes):
    plt.figure(figsize=(8, 6))
    plt.scatter(atributos[:, 0], atributos[:, 1], c=classes, cmap="coolwarm", edgecolors="k")
    plt.xlabel("Atributo 1")
    plt.ylabel("Atributo 2")
    plt.title("Distribuição das Classes - Spiral3D")
    plt.grid(True)
    plt.show()

plot_dados(atributos, classes)

# ==============================
# 4. Verificação de estrutura dos dados
# ==============================
print("\nFormato dos dados:", atributos.shape)
print("\nDistribuição das classes:")
print("Classe -1:", np.sum(classes == -1))
print("Classe 1:", np.sum(classes == 1))


# ==============================
# 1. Função para dividir os dados
# ==============================
def dividir_dados(atributos, classes, proporcao_treino=0.8):
    # Embaralhar os índices para garantir aleatoriedade
    indices = np.arange(len(atributos))
    np.random.shuffle(indices)

    # Definir quantidade de dados de treino
    n_treino = int(len(atributos) * proporcao_treino)

    # Separar os dados
    X_treino, X_teste = atributos[indices[:n_treino]], atributos[indices[n_treino:]]
    y_treino, y_teste = classes[indices[:n_treino]], classes[indices[n_treino:]]

    return X_treino, X_teste, y_treino, y_teste

# ==============================
# 2. Aplicar a divisão nos dados carregados
# ==============================
X_treino, X_teste, y_treino, y_teste = dividir_dados(atributos, classes)

# Verificar o tamanho dos conjuntos resultantes
print("\nTamanho do conjunto de treinamento:", X_treino.shape[0])
print("Tamanho do conjunto de teste:", X_teste.shape[0])




# ==============================
# 1. Classe Perceptron Simples com Teste de Hiperparâmetros
# ==============================
class Perceptron:
    def __init__(self, n_atributos, taxa_aprendizado=0.01, n_epocas=100, inicializacao="zeros"):
        if inicializacao == "zeros":
            self.pesos = np.zeros(n_atributos)
        elif inicializacao == "aleatoria":
            self.pesos = np.random.uniform(-0.01, 0.01, n_atributos)  # Inicialização aleatória pequena
        self.vies = 0
        self.taxa_aprendizado = taxa_aprendizado
        self.n_epocas = n_epocas

    def ativacao(self, x):
        """Função de ativação: retorna -1 ou 1."""
        return 1 if x >= 0 else -1

    def treinar(self, X_treino, y_treino):
        """Treina o modelo ajustando os pesos."""
        for _ in range(self.n_epocas):
            for i in range(len(X_treino)):
                entrada = np.dot(X_treino[i], self.pesos) + self.vies
                saida = self.ativacao(entrada)
                erro = y_treino[i] - saida

                # Atualização dos pesos e viés
                self.pesos += self.taxa_aprendizado * erro * X_treino[i]
                self.vies += self.taxa_aprendizado * erro

    def prever(self, X):
        """Faz a previsão para um conjunto de entradas."""
        return np.array([self.ativacao(np.dot(x, self.pesos) + self.vies) for x in X])

# ==============================
# 2. Teste com diferentes configurações
# ==============================
configuracoes = [
    {"taxa_aprendizado": 0.001, "n_epocas": 100, "inicializacao": "zeros"},
    {"taxa_aprendizado": 0.01, "n_epocas": 200, "inicializacao": "zeros"},
    {"taxa_aprendizado": 0.05, "n_epocas": 300, "inicializacao": "aleatoria"},
    {"taxa_aprendizado": 0.1, "n_epocas": 500, "inicializacao": "aleatoria"}
]

resultados = []

for config in configuracoes:
    perceptron = Perceptron(X_treino.shape[1], **config)
    perceptron.treinar(X_treino, y_treino)
    
    y_predito = perceptron.prever(X_teste)
    acuracia = np.mean(y_predito == y_teste) * 100
    
    resultados.append((config, acuracia))
    print(f"Acurácia do Perceptron Simples ({config}): {acuracia:.2f}%")


# ==============================
# 3. Exibir resultados comparativos
# ==============================
print("\nResultados dos testes:")
for config, acuracia in resultados:
    print(f"Taxa Aprendizado: {config['taxa_aprendizado']}, Épocas: {config['n_epocas']}, Inicialização: {config['inicializacao']} → Acurácia: {acuracia:.2f}%")




import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. Normalização dos dados
# ==============================
atributos = (atributos - atributos.mean(axis=0)) / atributos.std(axis=0)

# ==============================
# 2. Classe MLP Melhorado - 3 Camadas Ocultas
# ==============================
class MLP:
    def __init__(self, n_atributos, n_ocultos1=150, n_ocultos2=100, n_ocultos3=50, taxa_aprendizado=0.005, n_epocas=1200):
        # Inicializar pesos das camadas
        self.pesos_entrada = np.random.uniform(-0.1, 0.1, (n_atributos, n_ocultos1))
        self.pesos_ocultos1 = np.random.uniform(-0.1, 0.1, (n_ocultos1, n_ocultos2))
        self.pesos_ocultos2 = np.random.uniform(-0.1, 0.1, (n_ocultos2, n_ocultos3))
        self.pesos_ocultos3 = np.random.uniform(-0.1, 0.1, n_ocultos3)
        self.vies_oculto1 = np.zeros(n_ocultos1)
        self.vies_oculto2 = np.zeros(n_ocultos2)
        self.vies_oculto3 = np.zeros(n_ocultos3)
        self.vies_saida = 0
        self.taxa_aprendizado = taxa_aprendizado
        self.n_epocas = n_epocas

    def ativacao(self, x):
        """Função de ativação sigmoide."""
        return 1 / (1 + np.exp(-x))

    def derivada_ativacao(self, x):
        """Derivada da sigmoide."""
        return x * (1 - x)

    def treinar(self, X_treino, y_treino):
        """Treina o modelo via Backpropagation."""
        historico_erro = []

        for epoca in range(self.n_epocas):
            erro_total = 0

            for i in range(len(X_treino)):
                # Forward
                entrada_oculta1 = np.dot(X_treino[i], self.pesos_entrada) + self.vies_oculto1
                saida_oculta1 = self.ativacao(entrada_oculta1)

                entrada_oculta2 = np.dot(saida_oculta1, self.pesos_ocultos1) + self.vies_oculto2
                saida_oculta2 = self.ativacao(entrada_oculta2)

                entrada_oculta3 = np.dot(saida_oculta2, self.pesos_ocultos2) + self.vies_oculto3
                saida_oculta3 = self.ativacao(entrada_oculta3)

                entrada_final = np.dot(saida_oculta3, self.pesos_ocultos3) + self.vies_saida
                saida_final = self.ativacao(entrada_final)

                # Cálculo do erro
                erro = y_treino[i] - saida_final
                erro_total += erro ** 2

                # Backpropagation
                erro_saida = erro * self.derivada_ativacao(saida_final)
                erro_oculto3 = erro_saida * self.pesos_ocultos3 * self.derivada_ativacao(saida_oculta3)
                erro_oculto2 = np.dot(erro_oculto3, self.pesos_ocultos2.T) * self.derivada_ativacao(saida_oculta2)
                erro_oculto1 = np.dot(erro_oculto2, self.pesos_ocultos1.T) * self.derivada_ativacao(saida_oculta1)

                # Atualização dos pesos
                self.pesos_ocultos3 += self.taxa_aprendizado * erro_saida * saida_oculta3
                self.pesos_ocultos2 += self.taxa_aprendizado * np.outer(saida_oculta2, erro_oculto3)
                self.pesos_ocultos1 += self.taxa_aprendizado * np.outer(saida_oculta1, erro_oculto2)
                self.pesos_entrada += self.taxa_aprendizado * np.outer(X_treino[i], erro_oculto1)
                self.vies_saida += self.taxa_aprendizado * erro_saida
                self.vies_oculto3 += self.taxa_aprendizado * erro_oculto3
                self.vies_oculto2 += self.taxa_aprendizado * erro_oculto2
                self.vies_oculto1 += self.taxa_aprendizado * erro_oculto1

            historico_erro.append(erro_total / len(X_treino))

        return historico_erro

    def prever(self, X):
        """Faz a previsão para um conjunto de entradas."""
        saida_oculta1 = self.ativacao(np.dot(X, self.pesos_entrada) + self.vies_oculto1)
        saida_oculta2 = self.ativacao(np.dot(saida_oculta1, self.pesos_ocultos1) + self.vies_oculto2)
        saida_oculta3 = self.ativacao(np.dot(saida_oculta2, self.pesos_ocultos2) + self.vies_oculto3)
        saida_final = self.ativacao(np.dot(saida_oculta3, self.pesos_ocultos3) + self.vies_saida)
        return np.where(saida_final >= 0.5, 1, -1)


# ==============================
# 3. Inicializar e Treinar o Modelo
# ==============================
mlp = MLP(X_treino.shape[1], n_ocultos1=150, n_ocultos2=100, n_ocultos3=50, taxa_aprendizado=0.005, n_epocas=1200)
historico_erro = mlp.treinar(X_treino, y_treino)

# ==============================
# 4. Avaliação do Modelo
# ==============================
y_predito_treino = mlp.prever(X_treino)
y_predito_teste = mlp.prever(X_teste)

acuracia_treino = np.mean(y_predito_treino == y_treino)
acuracia_teste = np.mean(y_predito_teste == y_teste)

print(f"Treinamento → Acurácia: {acuracia_treino * 100:.2f}%")
print(f"Teste → Acurácia: {acuracia_teste * 100:.2f}%")

# ==============================
# 5. Exibir Curva de Aprendizado
# ==============================
plt.plot(historico_erro, label="Erro de Treinamento")
plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.title("Curva de Aprendizado")
plt.legend()
plt.show()



# Subdimensionado

# ==============================
# 1. Normalização dos dados
# ==============================
atributos = (atributos - atributos.mean(axis=0)) / atributos.std(axis=0)

# ==============================
# 2. Classe MLP Subdimensionado - Underfitting
# ==============================
class MLP_Subdimensionado:
    def __init__(self, n_atributos, n_ocultos1=50, taxa_aprendizado=0.005, n_epocas=800):
        # Inicializar pesos das camadas
        self.pesos_entrada = np.random.uniform(-0.1, 0.1, (n_atributos, n_ocultos1))
        self.pesos_ocultos1 = np.random.uniform(-0.1, 0.1, n_ocultos1)
        self.vies_oculto1 = np.zeros(n_ocultos1)
        self.vies_saida = 0
        self.taxa_aprendizado = taxa_aprendizado
        self.n_epocas = n_epocas

    def ativacao(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_ativacao(self, x):
        return x * (1 - x)

    def treinar(self, X_treino, y_treino):
        for _ in range(self.n_epocas):
            for i in range(len(X_treino)):
                entrada_oculta1 = np.dot(X_treino[i], self.pesos_entrada) + self.vies_oculto1
                saida_oculta1 = self.ativacao(entrada_oculta1)

                entrada_final = np.dot(saida_oculta1, self.pesos_ocultos1) + self.vies_saida
                saida_final = self.ativacao(entrada_final)

                erro = y_treino[i] - saida_final
                erro_saida = erro * self.derivada_ativacao(saida_final)
                erro_oculto1 = erro_saida * self.pesos_ocultos1 * self.derivada_ativacao(saida_oculta1)

                self.pesos_ocultos1 += self.taxa_aprendizado * erro_saida * saida_oculta1
                self.pesos_entrada += self.taxa_aprendizado * np.outer(X_treino[i], erro_oculto1)
                self.vies_saida += self.taxa_aprendizado * erro_saida
                self.vies_oculto1 += self.taxa_aprendizado * erro_oculto1

    def prever(self, X):
        saida_oculta1 = self.ativacao(np.dot(X, self.pesos_entrada) + self.vies_oculto1)
        saida_final = self.ativacao(np.dot(saida_oculta1, self.pesos_ocultos1) + self.vies_saida)
        return np.where(saida_final >= 0.5, 1, -1)


# ==============================
# 3. Inicializar e Treinar o Modelo
# ==============================
mlp_sub = MLP_Subdimensionado(X_treino.shape[1], n_ocultos1=50, taxa_aprendizado=0.005, n_epocas=800)
mlp_sub.treinar(X_treino, y_treino)

# ==============================
# 4. Avaliação do Modelo
# ==============================
y_predito_treino = mlp_sub.prever(X_treino)
y_predito_teste = mlp_sub.prever(X_teste)

acuracia_treino = np.mean(y_predito_treino == y_treino)
acuracia_teste = np.mean(y_predito_teste == y_teste)

print(f"MLP Subdimensionado → Treinamento: {acuracia_treino * 100:.2f}% | Teste: {acuracia_teste * 100:.2f}%")




#SUPERDIMENSIONADO

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 1. Normalização dos dados
# ==============================
atributos = (atributos - atributos.mean(axis=0)) / atributos.std(axis=0)

# ==============================
# 2. Classe MLP Superdimensionado - Overfitting
# ==============================
class MLP_Superdimensionado:
    def __init__(self, n_atributos, n_ocultos1=200, n_ocultos2=150, n_ocultos3=100, n_ocultos4=50, taxa_aprendizado=0.002, n_epocas=1500):
        # Inicializar pesos das camadas
        self.pesos_entrada = np.random.uniform(-0.1, 0.1, (n_atributos, n_ocultos1))
        self.pesos_ocultos1 = np.random.uniform(-0.1, 0.1, (n_ocultos1, n_ocultos2))
        self.pesos_ocultos2 = np.random.uniform(-0.1, 0.1, (n_ocultos2, n_ocultos3))
        self.pesos_ocultos3 = np.random.uniform(-0.1, 0.1, (n_ocultos3, n_ocultos4))
        self.pesos_ocultos4 = np.random.uniform(-0.1, 0.1, n_ocultos4)
        self.vies_oculto1 = np.zeros(n_ocultos1)
        self.vies_oculto2 = np.zeros(n_ocultos2)
        self.vies_oculto3 = np.zeros(n_ocultos3)
        self.vies_oculto4 = np.zeros(n_ocultos4)
        self.vies_saida = 0
        self.taxa_aprendizado = taxa_aprendizado
        self.n_epocas = n_epocas

    def ativacao(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_ativacao(self, x):
        return x * (1 - x)

    def treinar(self, X_treino, y_treino):
        """Treina o modelo via Backpropagation."""
        for _ in range(self.n_epocas):
            for i in range(len(X_treino)):
                # Forward
                entrada_oculta1 = np.dot(X_treino[i], self.pesos_entrada) + self.vies_oculto1
                saida_oculta1 = self.ativacao(entrada_oculta1)

                entrada_oculta2 = np.dot(saida_oculta1, self.pesos_ocultos1) + self.vies_oculto2
                saida_oculta2 = self.ativacao(entrada_oculta2)

                entrada_oculta3 = np.dot(saida_oculta2, self.pesos_ocultos2) + self.vies_oculto3
                saida_oculta3 = self.ativacao(entrada_oculta3)

                entrada_oculta4 = np.dot(saida_oculta3, self.pesos_ocultos3) + self.vies_oculto4
                saida_oculta4 = self.ativacao(entrada_oculta4)

                entrada_final = np.dot(saida_oculta4, self.pesos_ocultos4) + self.vies_saida
                saida_final = self.ativacao(entrada_final)

                # Erro
                erro = y_treino[i] - saida_final
                erro_saida = erro * self.derivada_ativacao(saida_final)
                erro_oculto4 = erro_saida * self.pesos_ocultos4 * self.derivada_ativacao(saida_oculta4)
                erro_oculto3 = np.dot(erro_oculto4, self.pesos_ocultos3.T) * self.derivada_ativacao(saida_oculta3)
                erro_oculto2 = np.dot(erro_oculto3, self.pesos_ocultos2.T) * self.derivada_ativacao(saida_oculta2)
                erro_oculto1 = np.dot(erro_oculto2, self.pesos_ocultos1.T) * self.derivada_ativacao(saida_oculta1)

                # Atualização dos pesos
                self.pesos_ocultos4 += self.taxa_aprendizado * erro_saida * saida_oculta4
                self.pesos_ocultos3 += self.taxa_aprendizado * np.outer(saida_oculta3, erro_oculto4)
                self.pesos_ocultos2 += self.taxa_aprendizado * np.outer(saida_oculta2, erro_oculto3)
                self.pesos_ocultos1 += self.taxa_aprendizado * np.outer(saida_oculta1, erro_oculto2)
                self.pesos_entrada += self.taxa_aprendizado * np.outer(X_treino[i], erro_oculto1)
                self.vies_saida += self.taxa_aprendizado * erro_saida
                self.vies_oculto4 += self.taxa_aprendizado * erro_oculto4
                self.vies_oculto3 += self.taxa_aprendizado * erro_oculto3
                self.vies_oculto2 += self.taxa_aprendizado * erro_oculto2
                self.vies_oculto1 += self.taxa_aprendizado * erro_oculto1

    def prever(self, X):
        saida_oculta1 = self.ativacao(np.dot(X, self.pesos_entrada) + self.vies_oculto1)
        saida_oculta2 = self.ativacao(np.dot(saida_oculta1, self.pesos_ocultos1) + self.vies_oculto2)
        saida_oculta3 = self.ativacao(np.dot(saida_oculta2, self.pesos_ocultos2) + self.vies_oculto3)
        saida_oculta4 = self.ativacao(np.dot(saida_oculta3, self.pesos_ocultos3) + self.vies_oculto4)
        saida_final = self.ativacao(np.dot(saida_oculta4, self.pesos_ocultos4) + self.vies_saida)
        return np.where(saida_final >= 0.5, 1, -1)


# ==============================
# 3. Inicializar e Treinar o Modelo
# ==============================
mlp_super = MLP_Superdimensionado(X_treino.shape[1], n_ocultos1=200, n_ocultos2=150, n_ocultos3=100, n_ocultos4=50, taxa_aprendizado=0.002, n_epocas=1500)
mlp_super.treinar(X_treino, y_treino)

# ==============================
# 4. Avaliação do Modelo
# ==============================
y_predito_treino = mlp_super.prever(X_treino)
y_predito_teste = mlp_super.prever(X_teste)

acuracia_treino = np.mean(y_predito_treino == y_treino)
acuracia_teste = np.mean(y_predito_teste == y_teste)

print(f"MLP Superdimensionado → Treinamento: {acuracia_treino * 100:.2f}% | Teste: {acuracia_teste * 100:.2f}%")
