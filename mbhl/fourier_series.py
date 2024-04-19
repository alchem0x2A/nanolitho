import numpy as np
import matplotlib.pyplot as plt

def fourier_series(theta, coefficients, N):
    result = coefficients[0]
    for n in range(1, N+1):
        result += coefficients[2*n-1] * np.cos(n * theta) + coefficients[2*n] * np.sin(n * theta)
    return result

def mutate_circle_fourier(coefficients_x, coefficients_y, N, num_points=100):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = fourier_series(theta, coefficients_x, N)
    y = fourier_series(theta, coefficients_y, N)
    return x, y

def normalize_coefficients(coefficients):
    # Find the first non-zero coefficient
    for c in coefficients:
        if np.abs(c) > 1e-6:
            phase = np.angle(c)
            break

    # Normalize coefficients by this phase
    return coefficients * np.exp(-1j * phase)

def mutate_circle_fourier(coefficients_x, coefficients_y, N, num_points=100):
    # Normalize coefficients to remove rotational equivalency
    coefficients_x = normalize_coefficients(coefficients_x)
    coefficients_y = normalize_coefficients(coefficients_y)

    theta = np.linspace(0, 2*np.pi, num_points)
    x = fourier_series(theta, coefficients_x, N).real
    y = fourier_series(theta, coefficients_y, N).real
    return x, y

# Rest of your code remains the same


# Example coefficients and usage
N = 5
coefficients_x = np.random.uniform(-0.5, 0.5, 2 * N + 1)
coefficients_y = np.random.uniform(-0.5, 0.5, 2 * N + 1)

x, y = mutate_circle_fourier(coefficients_x, coefficients_y, N)
plt.plot(x, y)
plt.axis('equal')
plt.show()
