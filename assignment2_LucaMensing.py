import math
import random
import matplotlib.pyplot as plt
from statistics import mean, stdev

# Parameters
random.seed(7)
N_POINTS = 400
W = 80             # Historical error window (paper: 8000)
W_SHORT = 10       # Short-term average window (paper: 10)
EPSILON = 0.3     # Paper's likelihood threshold parameter
WARMUP = 100       # Indexes before starting anomaly detection

# 1. Generate a synthetic data stream
data = [
    10 + 2 * math.sin(i * 2 * math.pi / 40)
    + random.gauss(0, 0.18)
    for i in range(N_POINTS)
]

# 2. Inject artificial anomalies
anomaly_indices = [random.randint(100, 400) for _ in range(6)]

for i in anomaly_indices:
    data[i] += 5 if i % 2 == 0 else -5

# 3. Store results
predictions = []
errors = []
likelihoods = []
detected = []

# Store all prediction-error scores for the rolling distribution
score_history = []

# 4. Process the stream
for t, value in enumerate(data):

    # Predict current value using previous observations
    if t == 0:
        prediction = value
    elif t == 1:
        prediction = data[t - 1]
    else:
        prediction = (
            data[t - 1] + (data[t - 1] - data[t - 2])
        )

    # Prediction error s_t
    s_t = abs(value - prediction)
    errors.append(s_t)
    predictions.append(prediction)
    score_history.append(s_t)

    # 5. Calculate anomaly likelihood using paper equations
    if len(score_history) < W_SHORT:
        likelihood = 0.0
    elif t < WARMUP:
        likelihood = 0.0
    else:
        # Use up to W recent error scores
        recent = score_history[-W:]

        # Equation (2): rolling mean mu_t
        mu_t = mean(recent)

        # Equation (3): rolling sample variance and standard deviation
        if len(recent) >= 2:
            sigma_t = stdev(recent)
        else:
            sigma_t = 0.0

        # Equation (5): short-term mean mu_t tilde
        short_recent = score_history[-W_SHORT:]
        mu_tilde = mean(short_recent)

        # Equation (4): L_t = 1 - Q((mu_tilde - mu_t) / sigma_t)
        if sigma_t < 1e-12:
            z = 0.0 if mu_tilde <= mu_t else 10.0
        else:
            z = (mu_tilde - mu_t) / sigma_t

        # Q(z) is the upper tail of the standard normal distribution
        Q = 0.5 * math.erfc(z / math.sqrt(2))
        likelihood = 1.0 - Q

    likelihoods.append(likelihood)

    # Equation (6): anomaly detected if L_t >= 1 - epsilon
    is_anomaly = (
        t >= WARMUP
        and len(score_history) >= W_SHORT
        and likelihood >= 1.0 - EPSILON
    )
    detected.append(is_anomaly)

# 6. Print results
found = [i for i, flag in enumerate(detected) if flag]

print("Streaming anomaly detection demo")
print("Injected anomalies:", anomaly_indices)
print("Detected anomalies:", found)
print(
    "Injected anomalies detected:",
    sum(i in found for i in anomaly_indices),
    "out of",
    len(anomaly_indices),
)
print("Number of false positives are: ", len(found) - len(anomaly_indices))

# 7. Plot data and likelihood
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})

axes[0].plot(data, label="Observed data")
axes[0].plot(predictions, label="Predictions", alpha=0.7)
axes[0].scatter(
    found,
    [data[i] for i in found],
    color="red",
    marker="x",
    label="Detected anomalies",
)
axes[0].set_ylabel("Value")
axes[0].set_title("Streaming Anomaly Detection")
axes[0].legend()
axes[0].grid(True, alpha=0.25)

axes[1].plot(likelihoods, label="Anomaly likelihood")
axes[1].axhline(
    1.0 - EPSILON,
    linestyle="--",
    label="Detection threshold",
)
axes[1].set_xlabel("Time step")
axes[1].set_ylabel("Likelihood")
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend(loc="lower left")
axes[1].grid(True, alpha=0.25)

plt.tight_layout()
plt.show()


