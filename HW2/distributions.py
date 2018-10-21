import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.figure()
    plt.title("Initial point distribution")
    plt.hist([random.gauss(-10, 10) for _ in range(100)])
    plt.savefig("latex/initial_point_dist.png")

    plt.figure()
    plt.title("Restart frequency distribution")
    plt.hist([random.randint(2, 50) for _ in range(100)])
    plt.savefig("latex/restart_frequency_dist.png")

    plt.figure()
    plt.title("Epsilon (abs) distribution")
    plt.hist([random.triangular(1e-14, 1e-7) for _ in range(100)])
    plt.savefig("latex/epsilon_dist.png")