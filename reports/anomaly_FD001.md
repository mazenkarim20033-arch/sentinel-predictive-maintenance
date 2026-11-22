# Anomaly detection, FD001 (generated)

Detectors: statistical (mean squared z against healthy profile) and Isolation Forest, fit on the first 30% of life of training assets; threshold = 99th percentile of scores on a separate healthy calibration slice. 5-fold grouped CV; held-out assets never touch fitting or thresholds.

## A. Degradation analysis on real trajectories (descriptive, no accuracy claim)

Alert = score above threshold for 3 consecutive cycles. There are no ground-truth anomaly labels in C-MAPSS, so this only shows how the scores behave as engines degrade.

| detector | assets | with a sustained alert | alert inside healthy phase | median lead (cycles before failure) | lead p10-p90 | mean Spearman(score, time) | clean-phase false-positive rate |
|---|---|---|---|---|---|---|---|
| statistical | 100 | 1.00 | 0.03 | 50 | 28-79 | 0.74 | 0.017 |
| isolation_forest | 100 | 1.00 | 0.02 | 51 | 30-90 | 0.74 | 0.015 |

Median score divided by threshold, by true RUL bin (above 1 means the score exceeds the alert level):

| detector | RUL 0-25 | RUL 25-50 | RUL 50-100 | RUL 100-150 | RUL 150-inf |
|---|---|---|---|---|---|
| statistical | 4.16 | 1.73 | 0.59 | 0.34 | 0.32 |
| isolation_forest | 1.26 | 1.10 | 0.90 | 0.83 | 0.82 |

## B. Controlled synthetic faults (SYNTHETIC, not real anomalies)

One signal is modified inside the healthy phase of a held-out engine, 10 random injections per engine and setting. Magnitude is in healthy standard deviations; duration 15 cycles (spike: 1 cycle). Per-step alerts, no persistence. Precision counts alerts outside the injected interval as false.

### statistical (clean healthy-phase false-positive rate 0.017)

| fault | magnitude | event detection rate | step recall | precision | median delay (cycles) |
|---|---|---|---|---|---|
| spike | 1 | 0.03 | 0.03 | 0.02 | 0.0 |
| spike | 2 | 0.04 | 0.04 | 0.03 | 0.0 |
| spike | 3 | 0.07 | 0.07 | 0.07 | 0.0 |
| spike | 5 | 0.42 | 0.42 | 0.29 | 0.0 |
| step | 1 | 0.16 | 0.02 | 0.27 | 4.0 |
| step | 2 | 0.26 | 0.03 | 0.38 | 4.0 |
| step | 3 | 0.42 | 0.08 | 0.58 | 3.0 |
| step | 5 | 0.93 | 0.44 | 0.89 | 1.0 |
| ramp | 1 | 0.14 | 0.02 | 0.24 | 4.0 |
| ramp | 2 | 0.17 | 0.02 | 0.26 | 6.0 |
| ramp | 3 | 0.27 | 0.03 | 0.37 | 9.0 |
| ramp | 5 | 0.65 | 0.12 | 0.68 | 11.0 |
| stuck | 1 | 0.13 | 0.01 | 0.21 | 4.0 |
| stuck | 2 | 0.14 | 0.02 | 0.25 | 3.0 |
| stuck | 3 | 0.14 | 0.01 | 0.21 | 4.5 |
| stuck | 5 | 0.13 | 0.01 | 0.21 | 4.0 |
| noise_burst | 1 | 0.18 | 0.02 | 0.27 | 5.0 |
| noise_burst | 2 | 0.42 | 0.05 | 0.48 | 5.0 |
| noise_burst | 3 | 0.80 | 0.13 | 0.71 | 3.0 |
| noise_burst | 5 | 0.99 | 0.33 | 0.86 | 1.0 |

### isolation_forest (clean healthy-phase false-positive rate 0.015)

| fault | magnitude | event detection rate | step recall | precision | median delay (cycles) |
|---|---|---|---|---|---|
| spike | 1 | 0.02 | 0.02 | 0.02 | 0.0 |
| spike | 2 | 0.03 | 0.03 | 0.03 | 0.0 |
| spike | 3 | 0.04 | 0.04 | 0.04 | 0.0 |
| spike | 5 | 0.05 | 0.05 | 0.05 | 0.0 |
| step | 1 | 0.20 | 0.02 | 0.31 | 4.0 |
| step | 2 | 0.27 | 0.03 | 0.37 | 4.5 |
| step | 3 | 0.33 | 0.04 | 0.43 | 4.0 |
| step | 5 | 0.35 | 0.05 | 0.47 | 4.0 |
| ramp | 1 | 0.16 | 0.02 | 0.25 | 6.0 |
| ramp | 2 | 0.19 | 0.02 | 0.27 | 6.0 |
| ramp | 3 | 0.23 | 0.03 | 0.34 | 7.0 |
| ramp | 5 | 0.29 | 0.03 | 0.39 | 8.0 |
| stuck | 1 | 0.13 | 0.01 | 0.21 | 5.0 |
| stuck | 2 | 0.14 | 0.02 | 0.24 | 5.0 |
| stuck | 3 | 0.12 | 0.01 | 0.21 | 4.5 |
| stuck | 5 | 0.12 | 0.01 | 0.20 | 4.0 |
| noise_burst | 1 | 0.18 | 0.02 | 0.27 | 5.0 |
| noise_burst | 2 | 0.25 | 0.03 | 0.35 | 5.0 |
| noise_burst | 3 | 0.30 | 0.03 | 0.40 | 4.0 |
| noise_burst | 5 | 0.33 | 0.04 | 0.44 | 3.0 |
