# predicao_fuzzy.py
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from skfuzzy import control as ctrl

# Carregar o modelo fuzzy
sistema_controle = joblib.load('modelo_fuzzy.joblib')
simulacao = ctrl.ControlSystemSimulation(sistema_controle)

# Função para prever a qualidade com o sistema fuzzy
def prever_qualidade(nivel_comida, nivel_servico):
    simulacao.input['comida'] = nivel_comida
    simulacao.input['servico'] = nivel_servico
    simulacao.compute()
    valor = simulacao.output['qualidade']
    if valor < 13:
        return 'baixa'
    elif valor < 19:
        return 'media'
    else:
        return 'alta'

# Dados de teste (sintéticos)
np.random.seed(0)
data = pd.DataFrame({
    'comida': np.random.randint(0, 11, 100),
    'servico': np.random.randint(0, 11, 100),
    'qualidade_esperada': np.random.choice(['baixa', 'media', 'alta'], 100)
})

# Obter previsões para o dataset
data['qualidade_prevista'] = data.apply(lambda row: prever_qualidade(row['comida'], row['servico']), axis=1)

# Calcular métricas
acuracia = accuracy_score(data['qualidade_esperada'], data['qualidade_prevista'])
f1 = f1_score(data['qualidade_esperada'], data['qualidade_prevista'], average='weighted')
matriz_confusao = confusion_matrix(data['qualidade_esperada'], data['qualidade_prevista'], labels=['baixa', 'media', 'alta'])

# Exibir métricas
print(f"Acurácia: {acuracia:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Matriz de Confusão:")
print(matriz_confusao)

# Visualização das métricas
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plotagem da matriz de confusão
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=['baixa', 'media', 'alta'], yticklabels=['baixa', 'media', 'alta'], ax=axs[0])
axs[0].set_xlabel('Qualidade Prevista')
axs[0].set_ylabel('Qualidade Esperada')
axs[0].set_title('Matriz de Confusão')

# Plotagem da acurácia e F1 Score
metrics_data = pd.DataFrame({'Métrica': ['Acurácia', 'F1 Score'], 'Valor': [acuracia, f1]})
metrics_data.plot(kind='bar', x='Métrica', y='Valor', legend=False, ax=axs[1], color=['#4CAF50', '#FF5722'])
axs[1].set_ylim(0, 1)
axs[1].set_ylabel('Pontuação')
axs[1].set_title('Acurácia e F1 Score')

plt.tight_layout()
plt.show()
