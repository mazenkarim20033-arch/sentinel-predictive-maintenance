# Anomaly detection, FD004 (generated)

Detectors: statistical (mean squared z against healthy profile) and Isolation Forest, fit on the first 30% of life of training assets; threshold = 99th percentile of scores on a separate healthy calibration slice. 5-fold grouped CV; held-out assets never touch fitting or thresholds.

## A. Degradation analysis on real trajectories (descriptive, no accuracy claim)

Alert = score above threshold for 3 consecutive cycles. There are no ground-truth anomaly labels in C-MAPSS, so this only shows how the scores behave as engines degrade.

| detector | assets | with a sustained alert | alert inside healthy phase | median lead (cycles before failure) | lead p10-p90 | mean Spearman(score, time) | clean-phase false-positive rate |
|---|---|---|---|---|---|---|---|
| statistical | 249 | 1.00 | 0.02 | 43 | 12-84 | 0.75 | 0.023 |
| isolation_forest | 249 | 1.00 | 0.02 | 53 | 25-102 | 0.73 | 0.025 |

Median score divided by threshold, by true RUL bin (above 1 means the score exceeds the alert level):

| detector | RUL 0-25 | RUL 25-50 | RUL 50-100 | RUL 100-150 | RUL 150-inf |
|---|---|---|---|---|---|
| statistical | 2.94 | 1.25 | 0.47 | 0.26 | 0.22 |
| isolation_forest | 1.20 | 1.09 | 0.92 | 0.83 | 0.81 |

## B. Controlled synthetic faults (SYNTHETIC, not real anomalies)

One signal is modified inside the healthy phase of a held-out engine, 10 random injections per engine and setting. Magnitude is in healthy standard deviations; duration 15 cycles (spike: 1 cycle). Per-step alerts, no persistence. Precision counts alerts outside the injected interval as false.

### statistical (clean healthy-phase false-positive rate 0.023)

| fault | magnitude | event detection rate | step recall | precision | median delay (cycles) |
|---|---|---|---|---|---|
| spike | 1 | 0.02 | 0.02 | 0.01 | 0.0 |
| spike | 2 | 0.02 | 0.02 | 0.01 | 0.0 |
| spike | 3 | 0.04 | 0.04 | 0.02 | 0.0 |
| spike | 5 | 0.16 | 0.16 | 0.09 | 0.0 |
| step | 1 | 0.10 | 0.02 | 0.17 | 4.0 |
| step | 2 | 0.12 | 0.02 | 0.18 | 4.0 |
| step | 3 | 0.17 | 0.04 | 0.28 | 3.0 |
| step | 5 | 0.44 | 0.17 | 0.65 | 1.0 |
| ramp | 1 | 0.10 | 0.02 | 0.17 | 3.0 |
| ramp | 2 | 0.11 | 0.02 | 0.18 | 4.0 |
| ramp | 3 | 0.13 | 0.02 | 0.19 | 5.0 |
| ramp | 5 | 0.28 | 0.05 | 0.35 | 11.0 |
| stuck | 1 | 0.09 | 0.02 | 0.17 | 4.0 |
| stuck | 2 | 0.09 | 0.02 | 0.18 | 4.0 |
| stuck | 3 | 0.09 | 0.02 | 0.17 | 5.0 |
| stuck | 5 | 0.09 | 0.02 | 0.17 | 3.0 |
| noise_burst | 1 | 0.11 | 0.02 | 0.18 | 3.0 |
| noise_burst | 2 | 0.19 | 0.03 | 0.25 | 4.0 |
| noise_burst | 3 | 0.45 | 0.07 | 0.42 | 4.0 |
| noise_burst | 5 | 0.91 | 0.21 | 0.70 | 2.0 |

### isolation_forest (clean healthy-phase false-positive rate 0.025)

| fault | magnitude | event detection rate | step recall | precision | median delay (cycles) |
|---|---|---|---|---|---|
| spike | 1 | 0.03 | 0.03 | 0.02 | 0.0 |
| spike | 2 | 0.03 | 0.03 | 0.02 | 0.0 |
| spike | 3 | 0.04 | 0.04 | 0.02 | 0.0 |
| spike | 5 | 0.04 | 0.04 | 0.02 | 0.0 |
| step | 1 | 0.14 | 0.02 | 0.19 | 4.0 |
| step | 2 | 0.16 | 0.03 | 0.23 | 3.0 |
| step | 3 | 0.19 | 0.04 | 0.27 | 3.0 |
| step | 5 | 0.20 | 0.04 | 0.30 | 3.0 |
| ramp | 1 | 0.13 | 0.02 | 0.18 | 4.0 |
| ramp | 2 | 0.14 | 0.02 | 0.19 | 4.0 |
| ramp | 3 | 0.16 | 0.03 | 0.21 | 5.0 |
| ramp | 5 | 0.17 | 0.03 | 0.24 | 5.0 |
| stuck | 1 | 0.12 | 0.02 | 0.18 | 4.0 |
| stuck | 2 | 0.11 | 0.02 | 0.17 | 4.0 |
| stuck | 3 | 0.12 | 0.02 | 0.18 | 4.0 |
| stuck | 5 | 0.11 | 0.02 | 0.17 | 3.0 |
| noise_burst | 1 | 0.15 | 0.03 | 0.20 | 4.0 |
| noise_burst | 2 | 0.16 | 0.03 | 0.23 | 3.0 |
| noise_burst | 3 | 0.18 | 0.03 | 0.25 | 3.0 |
| noise_burst | 5 | 0.19 | 0.04 | 0.28 | 3.0 |
