import functions as f
import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2


print(f.numerical_gradient(function_2, np.array([3.0,4.0])))
print(f.numerical_gradient(function_2, np.array([0.0,2.0])))
print(f.numerical_gradient(function_2, np.array([3.0,0.0])))

print(f.gradient_descent(function_2, np.array([3.0,4.0]), lr=0.1))
print(f.gradient_descent(function_2, np.array([0.0,2.0]), lr=0.1))
print(f.gradient_descent(function_2, np.array([3.0,0.0]), lr=0.1))
print(f.gradient_descent(function_2, np.array([-3.0,4.0]), lr=0.1))
