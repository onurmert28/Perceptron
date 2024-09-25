import numpy as np
import tkinter as tk
from tkinter import messagebox

# Perceptron sınıfı
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)

                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted

    def _activation_function(self, x):
        return np.where(x >= 0, 1, 0)

# Arayüzü başlatma
def run_perceptron():
    try:
        # Verileri al ve şekillendir
        input_data = [
            list(map(int, entry_x1.get().split(','))),
            list(map(int, entry_x2.get().split(',')))
        ]
        X = np.array(input_data)

        output_data = list(map(int, entry_y.get().split(',')))
        y = np.array(output_data)

        # Perceptron modelini oluştur
        model = Perceptron(learning_rate=0.1, n_iter=10)
        model.fit(X, y)

        # Tahmin yap
        predict_input = list(map(int, entry_predict.get().split(',')))
        prediction = model.predict(np.array([predict_input]))

        # Sonucu göster
        messagebox.showinfo("Tahmin", f"Sonuç: {prediction[0]}")
    except Exception as e:
        messagebox.showerror("Hata", f"Bir hata oluştu: {e}")

# Tkinter arayüzü
root = tk.Tk()
root.title("Perceptron Sınıflandırma")

# Girdi etiket ve giriş kutuları
tk.Label(root, text="X1 verileri (örn: 0,1):").grid(row=0, column=0)
entry_x1 = tk.Entry(root)
entry_x1.grid(row=0, column=1)

tk.Label(root, text="X2 verileri (örn: 1,0):").grid(row=1, column=0)
entry_x2 = tk.Entry(root)
entry_x2.grid(row=1, column=1)

tk.Label(root, text="Y verileri (örn: 0,1):").grid(row=2, column=0)
entry_y = tk.Entry(root)
entry_y.grid(row=2, column=1)

# Tahmin için giriş kutusu
tk.Label(root, text="Tahmin etmek istediğiniz veri (örn: 1,1):").grid(row=3, column=0)
entry_predict = tk.Entry(root)
entry_predict.grid(row=3, column=1)

# "Analiz Et" butonu
analyze_button = tk.Button(root, text="Analiz Et", command=run_perceptron)
analyze_button.grid(row=4, column=0, columnspan=2)

# Arayüzü çalıştır
root.mainloop()
