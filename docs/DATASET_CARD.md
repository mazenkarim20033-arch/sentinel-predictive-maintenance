# Dataset Card

## NASA C-MAPSS turbofan degradation (primary)

- **Source:** NASA Prognostics Center of Excellence repository, `CMAPSSData.zip` (https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data). Citation: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, Denver, 2008. The files are not redistributed in this repository (`data/raw/` is git-ignored).
- **Nature:** simulated, from a physics-based engine model with injected degradation and sensor noise. It is not measurement data from real engines.
- **Structure (verified on the downloaded files):** 26 whitespace-separated columns: engine id, cycle, 3 operating settings, 21 sensors; no missing values.

| Subset | Operating conditions | Fault modes | Train / test engines | Train rows | Test rows |
|---|---|---|---|---|---|
| FD001 | 1 | 1 (HPC degradation) | 100 / 100 | 20,631 | 13,096 |
| FD002 | 6 | 1 | 260 / 259 | 53,759 | 33,991 |
| FD003 | 1 | 2 (HPC, fan) | 100 / 100 | 24,720 | 16,596 |
| FD004 | 6 | 2 | 249 / 248 | 61,249 | 41,214 |

- **Labels:** training engines run to failure (RUL = last cycle - current cycle). Test engines are truncated before failure and come with one true RUL per engine (at the last observed cycle). True RUL at earlier test cycles is derived as final RUL plus cycles elapsed.
- **Documentation discrepancy:** NASA's `readme.txt` lists FD004 as 248 train / 249 test engines; the files contain 249 train / 248 test. This project uses the files.
- **Known properties found during analysis:** constant sensors differ by subset (FD001: s1, s5, s10, s16, s18, s19; FD003: s1, s5, s16, s18, s19; none in FD002/FD004 before per-regime normalisation); the 3 settings in FD002/FD004 form 6 clearly separable regimes (silhouette 0.997); the shortest test history is 19 cycles (FD004).
- **Intended use here:** developing and comparing RUL, warning, anomaly and uncertainty methods. Not for certifying any real maintenance policy.
- **Licence:** NASA catalogue states no licence terms; cite the source above.

## NASA Li-ion battery aging (second machine type, demonstration only)

- **Source:** NASA PCoE "Battery Data Set" archive (S3 mirror `https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip`, 209,708,670 bytes). The NASA catalogue page lists no files and the dashlink landing page was unreachable when checked. Not redistributed.
- **Contents:** 34 distinct 18650 cells (some files duplicated across archives) with charge, discharge and impedance cycles at different temperatures, currents and cutoff voltages. Many capacity series are unusable as run-to-failure data (very low starting capacity, zeros, spikes, early stops).
- **Used:** cells B0005, B0006, B0007, B0018 (24 C, 2 A discharge), one row per discharge cycle. End of life is a project-defined 1.5 Ah (25% fade), not NASA's 1.4 Ah criterion, chosen after inspecting the data so that all four cells qualify. 379 labelled rows.
- **Intended use here:** showing that the pipeline runs on a different machine type. There is no held-out test set and only four trajectories.
