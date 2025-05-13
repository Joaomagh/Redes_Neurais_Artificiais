# Relatório: Justificativa dos Hiperparâmetros

## 1. Introdução  
Nesta seção, detalhamos a escolha dos hiperparâmetros dos modelos **ADALINE** e **Perceptron de Múltiplas Camadas (MLP)** utilizados na primeira etapa do trabalho. Cada hiperparâmetro foi definido considerando princípios de aprendizado de máquina e testes experimentais realizados.

## 2. ADALINE  
O modelo ADALINE é uma rede neural **linear** usada para **regressão**. Seus hiperparâmetros foram escolhidos visando **estabilidade** e **boa convergência** do erro médio quadrático (MSE).

### Hiperparâmetros do ADALINE e justificativa:
- **Taxa de aprendizado (`eta = 0.001`)**  
  Um valor moderado foi escolhido para **evitar oscilações** na atualização dos pesos e garantir um aprendizado eficiente. Valores muito altos levam à **divergência** do modelo, enquanto valores muito baixos tornam o treinamento **excessivamente lento**.  

- **Número de épocas (`epochs = 100`)**  
  Definido para permitir a convergência do erro sem aumentar desnecessariamente o tempo de execução. O ADALINE, por ser **linear**, converge rapidamente, tornando **100 épocas suficientes** para ajustes adequados.  

- **Critério de parada (`tol = 1e-3`)**  
  O treinamento é interrompido caso o erro **não melhore significativamente** entre as épocas. Isso evita execuções desnecessárias quando o modelo já está **próximo da melhor solução**.  

- **Normalização Z-score**  
  Aplicada para **garantir que os valores da entrada estejam padronizados**, já que o ADALINE pode ser sensível à escala dos dados.  

### Conclusão sobre ADALINE  
Com essas escolhas, o modelo conseguiu um MSE **baixo e estável**, indicando que os hiperparâmetros foram adequados.

## 3. Perceptron de Múltiplas Camadas (MLP)  
O modelo MLP é uma rede **não linear**, capaz de capturar padrões complexos nos dados. Seus hiperparâmetros foram ajustados para **garantir um equilíbrio entre aprendizado eficiente e generalização**.

### Hiperparâmetros do MLP e justificativa
- **Arquitetura da Rede (`hidden_layers`)**  
  Foram testadas diferentes topologias:
  - **Subdimensionado (`hidden_layers=[2]`)** → Poucos neurônios, levando a **underfitting**.  
  - **Intermediário (`hidden_layers=[10]`)** → Número de neurônios suficiente para capturar padrões sem exagero.  
  - **Superdimensionado (`hidden_layers=[50, 50, 50]`)** → Muitos neurônios e camadas, levando a **overfitting**.  
  O modelo intermediário foi escolhido como **equilíbrio entre aprendizado e estabilidade**.  

- **Função de ativação (`sigmoide`)**  
  Escolhida porque **mantém valores dentro de um intervalo definido**, evitando explosão nos gradientes. Ajuda na propagação do erro durante o treinamento da rede.  

- **Taxa de aprendizado (`lr = 0.01`)**  
  Selecionada para **garantir ajustes progressivos nos pesos** sem oscilações extremas. Valores menores (`0.001`) foram testados, mas `0.01` garantiu convergência mais rápida sem instabilidade.  

- **Número de épocas (`epochs = 200`)**  
  Permitindo tempo suficiente para convergência sem risco de **overfitting prematuro**. Com menos épocas (`100`), o modelo intermediário não estabilizava adequadamente.  

### Conclusão sobre MLP  
Com essa configuração, a rede MLP demonstrou um **equilíbrio entre aprendizado e generalização**, evitando os extremos de underfitting e overfitting.

## 4. Conclusão  
Os hiperparâmetros de ADALINE e MLP foram escolhidos **com base na estabilidade, eficiência e precisão dos modelos**. O MSE obtido nas simulações mostrou que **os modelos funcionam corretamente**, validando as escolhas feitas.  

Com este relatório, a equipe pode **documentar a justificativa exigida na AV3** e garantir que todos os ajustes aplicados estejam bem fundamentados.
