import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import energy, ContinuousHopfieldNet

def create_cluster(center, num_points, std=0.1):
    """
    Generates a cluster of points around a given center.

    Args:
    - center (tuple): The (x, y) coordinates of the cluster center.
    - num_points (int): Number of points in the cluster.
    - std (float): Standard deviation of the noise.

    Returns:
    - Tensor of shape (num_points, 2) containing the clustered points.
    """
    center = torch.tensor(center, dtype=torch.float64)  # Convert center to tensor
    noise = torch.randn(num_points, 2, dtype=torch.float64) * std  # Gaussian noise
    return center + noise  # Shift noise around the center

def main():

    D = 2
    N = 20
    beta = 1
    normalize = True
    n_samples = 20
    nb_basis = 10
    num_iters = 100

    model = ContinuousHopfieldNet(beta, nb_basis, 1, 1000, "cpu")
    torch.random.manual_seed(42)

    which = [0, 3, 8, 9]
    nplots = len(which)

    fig, axes = plt.subplots(2, nplots, figsize=(20, 6),
                             constrained_layout=True)

    patterns = []
    queries = []

    for i in range(4):
        patterns.append(torch.randn(N, D, dtype=torch.float64))
        queries.append(torch.randn(D, 1, dtype=torch.float64))
    patterns[0] = torch.cat((create_cluster((-1, 1), 7), create_cluster((1, -1), 7), create_cluster((-1, -1), 6)), dim =0)
    patterns[0] = patterns[0] / torch.sqrt(torch.sum(patterns[0]*patterns[0], dim=1)).unsqueeze(1)
    patterns[1] = patterns[1] / torch.sqrt(torch.sum(patterns[1]*patterns[1], dim=1)).unsqueeze(1)
    x = torch.rand(N, dtype=torch.float64) * (2 * math.pi) - math.pi  # Random x in [-π, π]
    y = 2*x 
    patterns[2]= torch.stack((x, y), dim=1)
    x = torch.rand(N, dtype=torch.float64) * (2 * math.pi) - math.pi  # Random x in [-π, π]
    y = 2*torch.sin(x) 

    patterns[3]= torch.stack((x, y), dim=1)
    queries[0].zero_()
    # Create a linspace for angles between 0 and 2*pi
    theta = torch.linspace(0, 2 * math.pi, N)

    # Calculate the x and y coordinates for the unit circle
    x = torch.cos(theta)
    y = torch.sin(theta)
    y = torch.stack((x, y), dim=1)

    axes[1, 0].plot(y[:, 0], y[:, 1],  color='C2', linewidth=4)
    axes[1, 1].plot(y[:, 0], y[:, 1],  color='C2', linewidth=4)
    axes[1,2].plot(patterns[2][:, 0], patterns[2][:, 1],  color='C2', linewidth=4)
    x = torch.linspace(-math.pi, math.pi, 1000)
    y_ = 2*torch.sin(x) 
    y = torch.stack((x, y_), dim=1)
    axes[1, 3].plot(y[:, 0], y[:, 1],  color='C2', linewidth=4)
    for i in range(nplots):
        X = patterns[i]
        
        query = queries[i]
        xmin, xmax = X[:, 0].min(), X[:, 0].max()
        ymin, ymax = X[:, 1].min(), max([query[1].max(), X[:, 1].max()])

        xmin -= .1
        ymin -= .1
        xmax += .1
        ymax += .1

        xx = np.linspace(xmin, xmax, n_samples)
        yy = np.linspace(ymin, ymax, n_samples)

        mesh_x, mesh_y = np.meshgrid(xx, yy)

        Q = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
        Q = torch.from_numpy(Q)
    
        # cmap = 'OrRd_r'
        cmap = 'viridis'
        E1 = energy(beta, X, Q).reshape(*mesh_x.shape)
        axes[0,i].contourf(mesh_x, mesh_y, E1, cmap=cmap)
        E2 = energy(beta, X.float(), Q.float(), model).reshape(*mesh_x.shape)
        axes[1,i].contourf(mesh_x, mesh_y, E2, cmap=cmap)
        model = ContinuousHopfieldNet(beta, nb_basis, num_iters, 1000, "cpu")
        for k, method in enumerate(["discrete", "continuous"]):
            xis = torch.zeros((num_iters, D))
            xi = query
            if method == "discrete":
                for j in range(num_iters):
                    xis[j, :] = xi[:, 0]
                    p = torch.softmax(X.mm(xi)*beta, dim=0)
                    xi = X.T.mm(p)
            else:
                xis[0,:]= query.view(1,D)
                xi, xis_ = model(X.float(), query.float().T, return_contexts = True)
                xis = torch.cat([xis, xis_], dim = 0).squeeze(1)

            first_point = xis[0]
# Plot a marker at the first point to represent the circumference
            axes[k, i].scatter(first_point[0], first_point[1], marker='o',facecolors='none', s=175, edgecolors='C1', linewidths=5, label='$q_0$',)
            axes[k, i].plot(xis[0:, 0], xis[0:, 1],
                            lw=4,
                            marker='.', markersize=12,
                            color='C1',
                            label='$q_t$')
            axes[0, 0].set_ylabel("Discrete HNet.", fontsize=25)
            axes[1, 0].set_ylabel("Continuous HNet.", fontsize=25)
        for ax in axes[:, i]:
            
            ax.plot(X[:, 0], X[:, 1], 's', markerfacecolor='w',
                    markeredgecolor='k', markeredgewidth=1, markersize=8,
                    label='$x_i$')
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            # ax.set_xlim(xmin-.2, xmax+.2)
            # ax.set_ylim(ymin-.2, ymax+.2)
            ax.set_xticks(())
            ax.set_yticks(())



    axes[0, 0].legend(fontsize = 25)

    plt.savefig("contours.pdf")
    plt.show()


if __name__ == '__main__':
    main()

