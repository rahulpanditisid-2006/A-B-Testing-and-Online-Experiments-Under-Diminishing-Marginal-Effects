import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# =====================================================
# Reward Function
# =====================================================

def concave_f(x):
    return -(x - 0.6)**2 + 2

x_star = 0.6
f_star = concave_f(x_star)

# =====================================================
# Algorithm 2 Simulation
# =====================================================

def experiment1(T, noiselevel, interval):

    n_count = 0
    i_count = 0
    current_interval = interval.copy()

    regret = 0

    # -------------------------
    # Stage 1: Bisection Search
    # -------------------------

    while n_count + 2 * int(
        np.ceil(((3/2)**(4*i_count)) * (np.log(T))**2)
    ) <= T/3:

        i_count += 1

        sample_size = int(
            np.ceil(
                ((3/2)**(4*i_count))
                * (np.log(T))**2
            )
        )

        sample_left = (
            current_interval[0]
            + (1/3)*(current_interval[1]-current_interval[0])
        )

        sample_right = (
            current_interval[0]
            + (2/3)*(current_interval[1]-current_interval[0])
        )

        s1 = np.mean(
            np.random.normal(
                0,
                noiselevel,
                sample_size
            )
        )

        s2 = np.mean(
            np.random.normal(
                0,
                noiselevel,
                sample_size
            )
        )

        f1 = concave_f(sample_left) + s1
        f2 = concave_f(sample_right) + s2

        regret += (
            f_star - concave_f(sample_left)
        ) * sample_size

        regret += (
            f_star - concave_f(sample_right)
        ) * sample_size

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

        n_count += 2 * sample_size

    # -------------------------
    # Stage 2: Zero-order GD
    # -------------------------

    x = (
        current_interval[0]
        + current_interval[1]
    ) / 2

    m0 = 2
    j_count = 1

    while n_count + int(np.ceil(m0*j_count**2)) < T:

        h = j_count**(-0.5)

        m = int(np.ceil(m0*j_count**2))

        g_estimator = 0

        for _ in range(m):

            U = np.random.uniform(-1, 1)

            x_plus = np.clip(
                x + h*U,
                current_interval[0],
                current_interval[1]
            )

            x_minus = np.clip(
                x - h*U,
                current_interval[0],
                current_interval[1]
            )

            noisy_plus = (
                concave_f(x_plus)
                + np.random.normal(
                    0,
                    noiselevel
                )
            )

            noisy_minus = (
                concave_f(x_minus)
                + np.random.normal(
                    0,
                    noiselevel
                )
            )

            regret += (
                f_star - concave_f(x_plus)
            )

            regret += (
                f_star - concave_f(x_minus)
            )

            kernel = (15/4) * (5*U - 7*(U**3))

            g_estimator += (
                (noisy_plus - noisy_minus)
                * kernel
                / (2*h)
            )

        g_estimator /= m

        x = x + (1/j_count) * g_estimator

        x = np.clip(
            x,
            current_interval[0],
            current_interval[1]
        )

        n_count += 2*m
        j_count += 1

    # -------------------------
    # Exploitation Phase
    # -------------------------

    regret += (
        f_star - concave_f(x)
    ) * (T - n_count)

    result_estimator = (
        concave_f(x)
        + np.mean(
            np.random.normal(
                0,
                noiselevel,
                max(1, T-n_count)
            )
        )
    )

    return result_estimator, regret


# =====================================================
# Generate Regret Curve
# =====================================================

T_values = [
    1500,
    2000,
    5000,
    10000,
    20000,
    50000,
    80000
]

mean_regret = []
lower_ci = []
upper_ci = []

for T in T_values:

    regret_samples = []

    for _ in range(50):

        _, regret = experiment1(
            T=T,
            noiselevel=0.25,
            interval=[0,1]
        )

        regret_samples.append(regret)

    regret_samples = np.array(regret_samples)

    avg_regret = np.mean(regret_samples)

    ci_low, ci_high = st.t.interval(
        confidence=0.95,
        df=len(regret_samples)-1,
        loc=avg_regret,
        scale=st.sem(regret_samples)
    )

    mean_regret.append(avg_regret)
    lower_ci.append(ci_low)
    upper_ci.append(ci_high)

# =====================================================
# Plot
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(
    T_values,
    mean_regret,
    marker='o',
    linewidth=2,
    label='Average Regret'
)

plt.fill_between(
    T_values,
    lower_ci,
    upper_ci,
    alpha=0.3,
    label='95% Confidence Interval'
)

plt.xlabel('Time Horizon $T$')
plt.ylabel('Cumulative Regret')
plt.title('Regret Performance of Algorithm 2')

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    'regret_curve.png',
    dpi=300
)

plt.show()

plt.figure(figsize=(8,5))

plt.plot(
    np.sqrt(T_values),
    mean_regret,
    marker='o'
)

plt.xlabel(r'$\sqrt{T}$')
plt.ylabel('Average Regret')
plt.title(r'Regret vs $\sqrt{T}$')

plt.grid(True)
plt.show()