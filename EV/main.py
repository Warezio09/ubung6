# Automated check that you use the correct Python version
from sys import version_info

if version_info[0] < 3 or version_info[1] < 10:
    raise Exception("Must be using Python 3.10 or newer")
###########################################################
import numpy as np
import matplotlib.pyplot as plt
from ev import power_method

if __name__ == '__main__':
    c = np.array([[(1 / (100 * 10 ** -9)) + (1 / 10 ** -8), -1 / 10 ** -8],
                  [-1 / 10 ** -8, (1 / (47 * 10 ** -9)) + (1 / 10 ** -8)]])
    l = np.array([[10 ** -5, 0], [0, 22 * 10 ** -6]])
    l_inv = np.linalg.inv(l)
    A = l_inv @ c
    x0 = np.array([1, 0], dtype=complex)
    result = power_method(A, x0)
    n = result[0].size
    print("Der grösste Eigenewert ist: ", result[0][n - 1])
    print("Der eigene Vektor ist: ", result[1])
    print("Die Kreisfrequenz w ist gleich: ", np.sqrt(result[0][n - 1]))
    with open('beobachtungen.txt', 'a') as f:
        f.write("\n=== Aufgabe 6.3 ===\n")
        f.write(f"Matrix A:\n{A}\n\n")
        f.write(f"Approximierter Eigenwert (lambda): {result[0][n - 1]}\n")
        f.write(f"Zugehöriger Eigenvektor (x): {result[1]}\n")

    import numpy as np
    import matplotlib.pyplot as plt
    from ev import power_method

    A = np.array([[5, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 3]], dtype=complex)

    # referenzberechnung
    eigvals = np.linalg.eig(A)[0]
    lambda_ref = eigvals[np.argmax(np.abs(eigvals))]

    # startvektoren
    startvecs = [np.array([1, -1, 1], dtype=complex),
                 np.array([0, 0, 1], dtype=complex)]
    results = []
    for x0 in startvecs:
        lambdas, v = power_method(A, x0)
        results.append((lambdas[-1], v, len(lambdas)))

    # konsole
    print("=== Aufgabe 6.4 ===")
    print(f"Referenz-Eigenwert (numpy.linalg.eig): {lambda_ref}")
    for i, (l, v, iters) in enumerate(results, 1):
        print(f"\nStartvektor {i}: {startvecs[i - 1]}")
        print(f"Approximierter Eigenwert: {l}")
        print(f"Approximierter Eigenvektor: {v}")

    # plotten
    plt.figure()
    for i, x0 in enumerate(startvecs, 1):
        lambdas, v = power_method(A, x0)
        errors = np.abs(lambdas - lambda_ref)
        plt.semilogy(errors, marker='o' if i == 1 else 's', label=f'Startvektor {i}')
    plt.title('Fehlerverlauf der Potenzmethode')
    plt.xlabel('Iteration')
    plt.ylabel('|λ - λ_ref|')
    plt.legend()
    plt.grid(True)
    plt.savefig('Plots/Comparison_different_start_vectors.pdf')

    # ergebnisse schreiben
    with open('beobachtungen.txt', 'a') as f:
        f.write("\n=== Aufgabe 6.4 ===\n")
        f.write(f"Referenz-Eigenwert: {lambda_ref}\n")
        for i, (l, v, iters) in enumerate(results, 1):
            f.write(f"\nStartvektor {i}: {startvecs[i - 1]}\n")
            f.write(f"Eigenwert: {l}\n")
            f.write(f"Eigenvektor: {v}\n")
        f.write("\nDer erste Startvektor konvergiert schneller.\n")

    A_complex = np.array([
        [3, 1],
        [0, 3 + 1j]
    ], dtype=complex)

    # startvektor
    x0_complex = np.array([0, 1], dtype=complex)

    # potenzmethode
    lambdas, eigenvector = power_method(A_complex, x0_complex)

    # Konsole
    print("\n=== Aufgabe 6.5 ===")
    print("Matrix A:")
    print(A_complex)
    print(f"\nStartvektor: {x0_complex}")
    print(f"Approximierter Eigenwert: {lambdas[-1]}")
    print(f"Approximierter Eigenvektor: {eigenvector}")

    # beobachtungen
    with open('beobachtungen.txt', 'a', encoding='utf-8') as f:
        f.write("\n\n=== Aufgabe 6.5 ===")
        f.write("\nKomplexwertige Matrix A:\n")
        f.write(str(A_complex))
        f.write(f"\n\nStartvektor: {x0_complex}")
        f.write(f"\nApproximierter Eigenwert: {lambdas[-1]}")
        f.write(f"\nApproximierter Eigenvektor: {eigenvector}")
        f.write("\n\nDie Potenzmethode konvergiert auch für komplexwertige Matrizen.")
