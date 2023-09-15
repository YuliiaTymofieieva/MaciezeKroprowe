from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("rand.mums", header=None, sep='\s{2,}', names=range(3))

table_names = ["> 1p_2s_rand_100nt", "> 1p_2s_rand_100nt Reverse"]

groups = df[0].isin(table_names).cumsum()

tables = {g.iloc[0, 0]: g.iloc[1:] for k, g in df.groupby(groups)}

tables_new = {}
for key, value in tables.items():
    tables_new[key] = value.copy()

# Wybór tabeli "1p_2s_rand_100nt"
table = tables_new["> 1p_2s_rand_100nt"]
print(table)

# Wybór tabeli "1p_2s_rand_100nt Reverse"
table_reverse = tables_new["> 1p_2s_rand_100nt Reverse"]
print(table_reverse)

# Konwersja danych na liczby całkowite dla "1p_2s_rand_100nt"
x_values = table[0].astype(int)
y_values = table[1].astype(int)
num_points = table[2].astype(int)

# Konwersja danych na liczby całkowite dla "1p_2s_rand_100nt Reverse"
x_values_reverse = table_reverse[0].astype(int)
y_values_reverse = table_reverse[1].astype(int)
num_points_reverse = table_reverse[2].astype(int)

# Tworzenie macierzy z wartościami dla "1p_2s_rand_100nt"
matrix_data = []
for x, y, num in zip(x_values, y_values, num_points):
    for i in range(num):
        matrix_data.append([x + i, y + i])

# Tworzenie macierzy z wartościami dla "1p_2s_rand_100nt Reverse"
matrix_data_reverse = []
for x, y, num in zip(x_values_reverse, y_values_reverse, num_points_reverse):
    for i in range(num):
        matrix_data_reverse.append([x + i, y + i])

# Przekształcenie danych do macierzy numpy dla "1p_2s_rand_100nt"
matrix_data = np.array(matrix_data)

# Przekształcenie danych do macierzy numpy dla "1p_2s_rand_100nt Reverse"
matrix_data_reverse = np.array(matrix_data_reverse)

# Podział danych na kolumny dla "1p_2s_rand_100nt"
x = matrix_data[:, 0] - 1
y = matrix_data[:, 1] - 1

# Podział danych na kolumny dla "1p_2s_rand_100nt Reverse"
x_reverse = matrix_data_reverse[:, 0] - 1
y_reverse = matrix_data_reverse[:, 1] - 1

# Wartość maksymalna dla osi X i Y dla "1p_2s_rand_100nt"
max_x = int(np.max(x))
max_y = int(np.max(y))

# Wartość maksymalna dla osi X i Y dla "1p_2s_rand_100nt Reverse"
max_x_reverse = int(np.max(x_reverse))
max_y_reverse = int(np.max(y_reverse))

# Utworzenie macierzy dla "1p_2s_rand_100nt"
matrix = np.zeros((max_x + 1, max_y + 1))

# Utworzenie macierzy dla "1p_2s_rand_100nt Reverse"
matrix_reverse = np.zeros((max_x_reverse + 1, max_y_reverse + 1))

# Wypełnienie macierzy wartościami dla "1p_2s_rand_100nt"
for i in range(len(x)):
    matrix[x[i], y[i]] = 1

# Wypełnienie macierzy wartościami dla "1p_2s_rand_100nt Reverse"
for i in range(len(x_reverse)):
    matrix_reverse[x_reverse[i], y_reverse[i]] = 1

# Tworzenie wykresu dla "1p_2s_rand_100nt"
fig, ax = plt.subplots()
ax.imshow(matrix, cmap='binary', origin='lower', extent=[0, max_y+1, 0, max_x+1])
ax.scatter(y, x, facecolors='black', edgecolors='none', s=10)
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_title('Macierz - 1p_2s_rand_100nt')
ax.grid(color='black', linewidth=0.5)

# Zapis wykresu do pliku
plt.savefig('wykres_1p_2s_rand_100nt.png', bbox_inches='tight')
plt.close()

# Tworzenie wykresu dla "1p_2s_rand_100nt Reverse"
fig, ax = plt.subplots()
ax.imshow(matrix_reverse, cmap='binary', origin='lower', extent=[0, max_y_reverse+1, 0, max_x_reverse+1])
ax.scatter(y_reverse, x_reverse, facecolors='black', edgecolors='none', s=10)
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_title('Macierz - 1p_2s_rand_100nt Reverse')
ax.grid(color='black', linewidth=0.5)

# Zapis wykresu do pliku
plt.savefig('wykres_1p_2s_rand_100nt_reverse.png', bbox_inches='tight')
plt.close()