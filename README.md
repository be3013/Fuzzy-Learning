# Sistema Fuzzy para Avaliação de Qualidade de Serviço
https://colab.research.google.com/drive/1w7daBmb0JZIvWIe7pAMQUBahwXibXChR?usp=sharing

Este projeto utiliza um sistema de lógica fuzzy para avaliar a qualidade de um serviço com base em duas entradas: comida e serviço. O modelo é treinado com um conjunto de regras fuzzy e salva um arquivo serializado, que pode ser usado para fazer predições em um segundo script.

## Estrutura do Projeto
- treinamento_fuzzy.py: Script para criar e treinar o modelo fuzzy. Define as variáveis fuzzy, configura as regras fuzzy e salva o modelo em um arquivo.
- predicao_fuzzy.py: Script para carregar o modelo salvo e realizar predições de qualidade. Calcula métricas como acurácia e F1 Score e exibe uma matriz de confusão.

## Script de Treinamento (treinamento_fuzzy.py)
- Variáveis de Entrada: comida e servico, variando de 0 a 10.
- Variável de Saída: qualidade, variando de 0 a 25.
- Regras Fuzzy: São definidas regras de inferência para calcular a qualidade com base na avaliação de comida e serviço.
- Salvar Modelo: O sistema fuzzy treinado é salvo em modelo_fuzzy.joblib.

## Script de Predição (predicao_fuzzy.py)
- Carregamento do Modelo: Carrega o modelo salvo.
- Predições: Realiza previsões de qualidade de serviço com base nos dados de teste.
- Métricas: Calcula a acurácia e o F1 Score, além de gerar uma matriz de confusão.
- Visualização: Plota a matriz de confusão e gráficos de acurácia e F1 Score.

## Observações
- A precisão do modelo pode variar dependendo das regras fuzzy definidas e dos dados de teste, mas a versão feita possuiu uma acurácia baixa.
