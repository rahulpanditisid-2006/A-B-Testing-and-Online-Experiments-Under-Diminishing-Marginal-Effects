import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Reward function
# -------------------------

def concave_f(x):
    return -(x - 0.6)**2 + 2

# True optimum
x_star = 0.6
f_star = concave_f(x_star)

# -------------------------
# Algorithm 1 simulation
# -------------------------

def experiment(T, noiselevel, interval):

    n_count = 0
    i_count = 0
    current_interval = interval

    while n_count + 2 * int(np.ceil(((3/2)**(4*i_count)) * (np.log(T))**2)) <= T/4:

        i_count += 1

        sample_size = int(
            np.ceil(
                ((3/2)**(4*i_count))
                * (np.log(T))**2
            )
        )

        sample_left = current_interval[0] + (1/3)*(current_interval[1]-current_interval[0])

        sample_right = current_interval[0] + (2/3)*(current_interval[1]-current_interval[0])

        f1 = (
            concave_f(sample_left)
            + np.mean(
                np.random.normal(
                    0,
                    noiselevel,
                    sample_size
                )
            )
        )

        f2 = (
            concave_f(sample_right)
            + np.mean(
                np.random.normal(
                    0,
                    noiselevel,
                    sample_size
                )
            )
        )

        if f1 <= f2:
            current_interval = [
                sample_left,
                current_interval[1]
            ]
        else:
            current_interval = [
                current_interval[0],
                sample_right
            ]

        n_count += 2*sample_size

    midpoint = (
        current_interval[0]
        + current_interval[1]
    )/2

    final_noise = np.mean(
        np.random.normal(
            0,
            noiselevel,
            T - n_count
        )
    )

    mu_hat = concave_f(midpoint) + final_noise

    return np.sqrt(T)*(mu_hat - f_star)

# -------------------------
# Run 500 experiments
# -------------------------

T = 500000
noiselevel = 0.25
Nrep = 5000

results = []

for _ in range(Nrep):
    results.append(
        experiment(
            T,
            noiselevel,
            [0,1]
        )
    )

results = np.array(results)

# -------------------------
# Plot
# -------------------------

plt.figure(figsize=(8,5))

plt.hist(
    results,
    bins=40,
    density=True,
    alpha=0.75
)

mu = np.mean(results)
sigma = np.std(results)

x = np.linspace(
    results.min(),
    results.max(),
    1000
)

pdf = (
    np.exp(
        -(x-mu)**2/(2*sigma**2)
    )
    /
    (sigma*np.sqrt(2*np.pi))
)

plt.plot(
    x,
    pdf,
    linewidth=2,
    label='Fitted Normal Density'
)

plt.axvline(
    0,
    linestyle='--',
    linewidth=2,
    label='Zero'
)

plt.xlabel(
    r'$\sqrt{T}(\hat{\mu}_f^{(T)}-f(x^*))$'
)

plt.ylabel('Density')

plt.title(
    'Bias of Algorithm 1'
)

plt.legend()

plt.tight_layout()

plt.savefig(
    'alg1_bias.png',
    dpi=300
)

plt.show()

print("Mean =", mu)
print("Std =", sigma)