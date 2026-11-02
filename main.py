import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import accuracy_score,precision_score,mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
df = pd.read_csv(url)

print('Первые строки таблицы импорта')
print(df.head(5))

print('Баланс классов:')
print(df['species'].value_counts())

df['target'] = df['species'].map({'setosa': 0, 'versicolor':1, 'virginica':2})

print('Обновленная таблица')
print(df.head(5))

# Признаки
x = df[['sepal_length',  'sepal_width',  'petal_length',  'petal_width']].values


for i in ['setosa', 'versicolor', 'virginica']:
   print(i + '\n')
   subset = df[df['species'] == i]
   for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
      print(f'{col}, min -{subset[col].min()}-, max -{subset[col].max()}-')
    



# Ответы
y = df['target'].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

print(f"Обучающих примеров {len(x_train)}, тестовых примеров {len(x_test)}")


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


#настройка

def manual_rule_predict(row):
   if row[2] < 2.5: return 0
   elif row[2] < 5: return 1
   else: return 2


manual_preds = [manual_rule_predict(x) for x in x_test]
manual_acc = accuracy_score(y_test, manual_preds)

print(f'Точность {manual_acc * 100:.1f}%')

df_manual = pd.DataFrame(x_test, columns=['sepal_length',  'sepal_width',  'petal_length',  'petal_width'])

df_manual['predicted_class'] = manual_preds
df_manual.to_csv('manual_rules.csv', index=False)
print('Файл сохранен')

# Обучение деревом решений
print('Результаты дерева решений')
depths = [3,5,7,10]

for d in depths:
   model = DecisionTreeClassifier(max_depth=d)

   model.fit(x_train,y_train)

   y_pred = model.predict(x_test)

   acc = accuracy_score(y_test, y_pred)

   prec = precision_score(y_test, y_pred, average='weighted')

   print(f"глубина {d}: точностб = {acc:.3f}, качество = {prec:.3f}")


# Обучение линейной регрессией

iterations = list(range(10, 101, 10))


reg = SGDRegressor(max_iter=1, warm_start=True, learning_rate='constant', eta0=0.01)

weights_his = []
loss_his = []
acc_his= []

for i, it in enumerate(iterations):
   if i==0:
      reg.max_iter = it
   else:
      reg.max_iter = it - iterations[i-1]

   reg.fit(x_train_scaled, y_train)

   w = np.append(reg.coef_, reg.intercept_)
   weights_his.append(w)

   y_pred_cont = reg.predict(x_test_scaled)

   loss = mean_squared_error(y_test, y_pred_cont)
   loss_his.append(loss)



   y_pred_class = np.round(y_pred_cont).clip(0, 2).astype(int)
   acc = accuracy_score(y_test, y_pred_class)
   acc_his.append(acc)

   print(f"Итераций {it}: Ошибка = {loss:.3f}, Точность = {acc:.3f}")

x_all_scaled = scaler.transform(x)

x_all_scaled = scaler.transform(x)
y_all_pred_cont = reg.predict(x_all_scaled)
y_all_pred_class = np.round(y_all_pred_cont).clip(0, 2).astype(int)

df_all_reg = pd.DataFrame(x, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df_all_reg['true_class'] = y
df_all_reg['predicted_class'] = y_all_pred_class

# Сохраняем
df_all_reg.to_csv('regression_predictions_full.csv', index=False)
print("Сохранено")


# ---- Считаем L2-нормы ----
l2_weights = [np.linalg.norm(w) for w in weights_his]

l2_updates = [0]  # для первой итерации изменения нет
for i in range(1, len(weights_his)):
    diff = np.linalg.norm(weights_his[i] - weights_his[i-1])
    l2_updates.append(diff)

# ---- Создаём фигуру с 4 графиками ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# График (a): Loss
axes[0, 0].plot(iterations, loss_his, marker='o', color='red', linewidth=2)
axes[0, 0].set_title('a) График потерь (Loss)')
axes[0, 0].set_xlabel('Итерации')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].grid(True)

# График (b): Accuracy
axes[0, 1].plot(iterations, acc_his, marker='s', color='blue', linewidth=2)
axes[0, 1].set_title('b) Точность (Accuracy)')
axes[0, 1].set_xlabel('Итерации')
axes[0, 1].set_ylabel('Точность')
axes[0, 1].grid(True)

# График (c): L2-норма весов
axes[1, 0].plot(iterations, l2_weights, marker='d', color='purple', linewidth=2)
axes[1, 0].set_title('c) L2-норма вектора весов')
axes[1, 0].set_xlabel('Итерации')
axes[1, 0].set_ylabel('||w||₂')
axes[1, 0].grid(True)

# График (d): L2-норма изменения весов
axes[1, 1].plot(iterations, l2_updates, marker='x', color='orange', linewidth=2)
axes[1, 1].set_title('d) L2-норма изменения весов')
axes[1, 1].set_xlabel('Итерации')
axes[1, 1].set_ylabel('||Δw||₂')
axes[1, 1].grid(True)

# Настраиваем общий вид и сохраняем
plt.tight_layout()
plt.savefig('my_graphs.png', dpi=300)
plt.show()

print("Графики сохранены в 'my_graphs.png'")