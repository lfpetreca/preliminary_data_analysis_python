import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

tabela = pd.read_csv('./advertising.csv')
print(tabela)

sns.heatmap(tabela.corr(), cmap='Wistia', annot=True)
plt.show()

sns.pairplot(tabela)
plt.show()

# separar as informações em X e Y
y = tabela['Vendas']
x = tabela.drop('Vendas', axis=1)
# aplicar o train test slipt
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

modelo_regressao_linear = LinearRegression()
modelo_randomforest = RandomForestRegressor()

modelo_regressao_linear.fit(x_treino, y_treino)
modelo_randomforest.fit(x_treino, y_treino)

previsao_regressao_linear = modelo_regressao_linear.predict(x_teste)
previsao_randomforest = modelo_randomforest.predict(x_teste)

print(metrics.r2_score(y_teste,previsao_regressao_linear))
print(metrics.r2_score(y_teste,previsao_randomforest))

# RandomFores é o melhor modelo 
tabela_aux = pd.DataFrame()
tabela_aux['y_teste'] = y_teste
tabela_aux['linear-regression'] = previsao_regressao_linear
tabela_aux['random-forest'] = previsao_randomforest

plt.figure(figsize=(15, 5))
sns.lineplot(data=tabela_aux)
plt.show()

sns.barplot(x=x_treino.columns, y=modelo_randomforest.feature_importances_)
plt.show()