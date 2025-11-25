from cost_function import cost_function
from gradient_descent import gradient_descent
import matplotlib.pyplot as plt

def main():
    w = 0.0 # weight
    b = 0.0 # bias
    a = 0.001 # learning rate
    n = 1000 # gd iterations

    points = [
        (1, 2), (2, 2.5), (3, 3.7), (4, 4.1),
        (5, 5.3), (6, 5.8), (7, 7.2), (8, 7.9),
        (9, 9.0), (10, 10.5), (11, 10.9), (12, 12.2),
        (13, 13.1), (14, 14.0), (15, 15.5)
    ]

    # compute gradient descent path
    w_values = [w]
    b_values = [b]
    for _ in range(n):
        w, b = gradient_descent(w, b, a, points)
        w_values.append(w)
        b_values.append(b)

    n_grid = 30 
    w_min, w_max = min(w_values), max(w_values)
    b_min, b_max = min(b_values), max(b_values)
    w_grid = [w_min + i*(w_max-w_min)/(n_grid-1) for i in range(n_grid)]
    b_grid = [b_min + i*(b_max-b_min)/(n_grid-1) for i in range(n_grid)]

    X, Y, Z = [], [], []
    for wi in w_grid:
        for bk in b_grid:
            X.append(wi)
            Y.append(bk)
            Z.append(cost_function(wi, bk, points))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, color='cyan', alpha=0.6, edgecolor='gray')

    # gradient descent trajectory
    J_traj = [cost_function(w_values[i], b_values[i], points) for i in range(len(w_values))]
    ax.plot(w_values, b_values, J_traj, color='red', marker='o', label='Gradient Descent Path')

    # labels
    ax.set_xlabel('w')
    ax.set_ylabel('b')
    ax.set_zlabel('J(w, b)')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
