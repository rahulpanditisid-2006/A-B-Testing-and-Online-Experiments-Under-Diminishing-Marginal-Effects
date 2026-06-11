import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# =====================================================
# Reward Function
# =====================================================

def concave_f(x):
    return -(x - 0.6)**2 + 2

x_star = 0.6
f_star = concave_f(x_star)

# =====================================================
# Algorithm 2
# =====================================================

def algorithm2(T, sigma=0.25):

    # -------------------------
    # Stage 1: Bisection
    # -------------------------

    interval = [0.0, 1.0]

    gamma1 = 1/6
    gamma2 = 2/3

    budget1 = int(gamma1 * T)

    used = 0
    k = 1

    while used < budget1:

        Nk = int(np.ceil((1.5)**(4*k) * np.log(T)**2))

        if used + 2*Nk > budget1:
            break

        left = interval[0] + (interval[1]-interval[0])/3
        right = interval[0] + 2*(interval[1]-interval[0])/3

        y_left = concave_f(left) + np.mean(
            np.random.normal(0,sigma,Nk)
        )

        y_right = concave_f(right) + np.mean(
            np.random.normal(0,sigma,Nk)
        )

        if y_left <= y_right:
            interval[0] = left
        else:
            interval[1] = right

        used += 2*Nk
        k += 1

    # -------------------------
    # Stage 2: Zeroth-order GD
    # -------------------------

    budget2 = int(gamma2*T)

    x = (interval[0]+interval[1])/2

    c0 = 1

    spent = 0
    i = 1

    while spent < budget2:

        h = i**(-0.5)

        m = int(np.ceil(i**2))

        if spent + 2*m > budget2:
            break

        grad_est = 0

        for _ in range(m):

            U = np.random.uniform(-1,1)

            xp = np.clip(
                x + h*U,
                interval[0],
                interval[1]
            )

            xm = np.clip(
                x - h*U,
                interval[0],
                interval[1]
            )

            fp = (
                concave_f(xp)
                + np.random.normal(0,sigma)
            )

            fm = (
                concave_f(xm)
                + np.random.normal(0,sigma)
            )

            K = (15/4)*(5*U - 7*(U**3))

            grad_est += (
                (fp-fm)*K/(2*h)
            )

        grad_est /= m

        x = x + (c0/i)*grad_est

        x = np.clip(
            x,
            interval[0],
            interval[1]
        )

        spent += 2*m
        i += 1

    # -------------------------
    # Stage 3: Inference
    # -------------------------

    remaining = T - used - spent

    rewards = (
        concave_f(x)
        +
        np.random.normal(
            0,
            sigma,
            remaining
        )
    )

    mu_hat = np.mean(rewards)

    regret = (
        f_star - concave_f(x)
    ) * T

    return mu_hat, x, regret

# =====================================================
# CLT Experiment
# =====================================================

T = 50000
Nrep = 500

vals = []

for _ in range(Nrep):

    mu_hat, _, _ = algorithm2(T)

    vals.append(
        np.sqrt(T)*(mu_hat-f_star)
    )

vals = np.array(vals)

mu = np.mean(vals)
sd = np.std(vals)

x = np.linspace(
    vals.min(),
    vals.max(),
    1000
)

plt.figure(figsize=(8,5))

plt.hist(
    vals,
    bins=30,
    density=True,
    alpha=0.7
)

plt.plot(
    x,
    norm.pdf(x,mu,sd),
    linewidth=2,
    color='red'
)

plt.axvline(
    0,
    linestyle='--',
    color='black'
)

plt.xlabel(
    r'$\sqrt{T}(\hat\mu_f-f(x^*))$'
)

plt.ylabel("Density")

plt.title(
    "Algorithm 2 CLT Verification"
)

plt.tight_layout()

plt.savefig(
    "alg2_clt.png",
    dpi=300
)

plt.show()

print("Mean =",mu)
print("Std =",sd)