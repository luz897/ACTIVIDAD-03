import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Definimos la matriz aumentada (A|b) del sistema de ecuaciones
A = np.array([
    [-2, 3, 5],
    [0, 3, 2],
    [0, 0, 2]
])

b = np.array([7, 0, 6])

# Resolviendo el sistema de ecuaciones usando sustitución hacia atrás
def back_substitution(A, b):
    n = len(b)
    x = np.zeros(n)
    
    # Comenzamos con la última ecuación
    x[n-1] = b[n-1] / A[n-1, n-1]
    
    # Realizamos la sustitución hacia atrás
    for i in range(n-2, -1, -1):
        sum_ax = 0
        for j in range(i+1, n):
            sum_ax += A[i, j] * x[j]
        x[i] = (b[i] - sum_ax) / A[i, i]
    
    return x

x_values = back_substitution(A, b)

st.write(f"Solución del sistema:")
st.write(f"x1 = {x_values[0]:.2f}")
st.write(f"x2 = {x_values[1]:.2f}")
st.write(f"x3 = {x_values[2]:.2f}")

def f1(x):
    return x**2 + x + 1

def f2(x):
    return x**2

x = np.linspace(0, 10, 400)
y1 = f1(x)
y2 = f2(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y1, label=r'$x^2 + x + 1$', color='purple')
plt.plot(x, y2, label=r'$x^2$', color='green', linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Comparación entre $x^2 + x + 1$ y $x^2$')
plt.legend()

st.pyplot(plt)

