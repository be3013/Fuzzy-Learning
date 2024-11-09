import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import joblib

# Definir variáveis fuzzy de entrada e saída
comida = ctrl.Antecedent(np.arange(0, 11, 1), 'comida')
servico = ctrl.Antecedent(np.arange(0, 11, 1), 'servico')
qualidade = ctrl.Consequent(np.arange(0, 26, 1), 'qualidade')

# Funções de pertinência
comida['ruim'] = fuzz.trimf(comida.universe, [0, 0, 5])
comida['media'] = fuzz.trimf(comida.universe, [0, 5, 10])
comida['boa'] = fuzz.trimf(comida.universe, [5, 10, 10])

servico['ruim'] = fuzz.trimf(servico.universe, [0, 0, 5])
servico['media'] = fuzz.trimf(servico.universe, [0, 5, 10])
servico['bom'] = fuzz.trimf(servico.universe, [5, 10, 10])

qualidade['baixa'] = fuzz.trimf(qualidade.universe, [0, 0, 13])
qualidade['media'] = fuzz.trimf(qualidade.universe, [0, 13, 25])
qualidade['alta'] = fuzz.trimf(qualidade.universe, [13, 25, 25])

# Regras fuzzy
regras = [
    ctrl.Rule(comida['ruim'] & servico['ruim'], qualidade['baixa']),
    ctrl.Rule(comida['ruim'] & servico['media'], qualidade['baixa']),
    ctrl.Rule(comida['ruim'] & servico['bom'], qualidade['media']),
    ctrl.Rule(comida['media'] & servico['ruim'], qualidade['baixa']),
    ctrl.Rule(comida['media'] & servico['media'], qualidade['media']),
    ctrl.Rule(comida['media'] & servico['bom'], qualidade['alta']),
    ctrl.Rule(comida['boa'] & servico['ruim'], qualidade['media']),
    ctrl.Rule(comida['boa'] & servico['media'], qualidade['alta']),
    ctrl.Rule(comida['boa'] & servico['bom'], qualidade['alta'])
]

# Sistema de controle
sistema_controle = ctrl.ControlSystem(regras)

# Salvar o modelo
joblib.dump(sistema_controle, 'modelo_fuzzy.joblib')
print("Modelo fuzzy salvo em 'modelo_fuzzy.joblib'")
