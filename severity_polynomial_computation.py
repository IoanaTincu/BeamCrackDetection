import numpy as np


# Polynomial coefficients
def severity(x):
    return (
            0.3223 * x ** 4
            - 0.0978 * x ** 3
            + 0.1078 * x ** 2
            - 0.0001 * x
            + 1e-6
    )


# 20 suggested normalized crack depths (excluding x = 0)
x_values = np.linspace(0.0, 0.38, 21)[1:]

# Compute severity values
severity_values = severity(x_values)

# Display results
print("x values:")
print(x_values)

print("\nSeverity values:")
print(", ".join(f"{s:.6e}" for s in severity_values))
