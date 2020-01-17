import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score

norm = MinMaxScaler()
dados = pd.read_csv('d:/basedados/urucui.csv')

#Carregando os dados do cenário
Y = dados['classe'].values
X = dados.drop(['classe'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.01, test_size=0.99, stratify=Y)
X_train = norm.fit_transform(X_train)
X_test = norm.transform(X_test)
acuracia = []

#Variando a quantidade de árvores
for i in np.arange(1,11,1):
    print('...... Vizinhos ', i)
    modelo = KNeighborsClassifier(n_neighbors=i)
    RF = RandomForestClassifier(n_estimators=i)
    RF.fit(X_train, y_train)
    preditas = RF.predict(X_test)
    a = accuracy_score(y_test, preditas)
    acuracia.append(a)


x = np.arange(0, 10)
plt.rcParams['figure.figsize'] = (14,8)
#fig, axs = plt.subplots(1, 2)
#fig.subplots_adjust(left=0.075, bottom=0.12, right=0.98, top=0.9, wspace=0.22, hspace=0.45)
plt.plot(x, acuracia, color = 'red', label='Acurácia')
#plt.title('Random Forest')
plt.xlabel('Quantidade de Vizinhos', fontsize=14)
plt.ylabel('Acurácia', fontsize=14)
plt.ylim(0.5, 1, 0.1)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend()
plt.grid(True)
plt.savefig('grafico_rf.png', dpi=300)
plt.show()