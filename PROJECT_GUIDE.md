# Jal Sakhi — Demo, Justification, and Team Guide

## Why simulation is valid

This project evaluates a network protocol and its reliability behavior. It does **not** claim measured LoRa range, battery lifetime, RF interference, or real-world sensor calibration. The CSV is a repeatable source of timestamped sensor readings. Each row is scored, packaged with a node ID and sequence number, sent through a configurable lossy link with retries, and acknowledged by a gateway.

The design is bridgeable because only the data-source and radio-adapter layers change when hardware is introduced:

| Simulation component | Hardware replacement | Contract retained |
| --- | --- | --- |
| CSV reader | ESP32 sensor acquisition | timestamp, TDS, turbidity, pH, temperature, level |
| Python scoring replay | `scoring.h` and `model_params.h` on ESP32 | feature order, threshold, verdict and reason bits |
| simulated loss/retries | LoRa driver send, timeout and retry loop | node ID, sequence number, retry policy |
| HTTP JSON post | LoRa gateway adapter forwarding normalized packets | packet fields and acknowledgement |
| local HTTP gateway | deployed gateway service | deduplication and unsafe-reading alert creation |

Viva statement: *“We use CSV replay to test the protocol deterministically and repeatably. With hardware, we replace only the sensor-input and radio-adapter layers; packet format, scoring model, sequence numbers, retry policy, and gateway contract remain unchanged.”*

## Run the project on Windows

Requirements: Python 3.10+ and GCC (MinGW is sufficient). No Python packages are required.

Open two PowerShell terminals in this folder.

Terminal 1:

```powershell
python gateway.py
```

Terminal 2:

```powershell
.\build.ps1
python simulator.py --limit 200 --loss 0.10 --retries 2 --report demo_report.json
```

The build produces `dataset.csv`, `model_params.h`, and `metrics.json`. The demo produces `demo_report.json`; the gateway records accepted packets in `gateway_events.jsonl`.

## What to show in the demo

1. Terminal 1 displays `Gateway listening on http://127.0.0.1:5000`.
2. Terminal 2 prints the experiment totals.
3. Open `demo_report.json`. It includes configuration, network model, packet/drop/delivery counts, first-attempt successes, retry recoveries, mean attempts per reading, detection results against CSV labels, five unsafe packet examples, and gateway totals.
4. Open `http://127.0.0.1:5000/summary`. Its `packets_received` should equal `delivered` in the report.

Say: “With a known 10% per-attempt loss, retries increase end-to-end delivery. The cost is visible as increased mean attempts per reading. The gateway accepts sequence-numbered packets, deduplicates them, and creates alerts for unsafe readings.”

## Network experiments

Use a new report file for each experiment:

```powershell
# Baseline: no loss and no retries.
python simulator.py --limit 200 --loss 0 --retries 0 --report baseline.json

# Loss without recovery.
python simulator.py --limit 200 --loss 0.25 --retries 0 --report loss_no_retry.json

# Same loss with recovery through retries.
python simulator.py --limit 200 --loss 0.25 --retries 2 --report loss_with_retry.json

# Persistent outage: retries cannot recover every 20th packet.
python simulator.py --limit 200 --loss 0.05 --retries 2 --outage-every 20 --report outage.json
```

Compare `delivery_rate`, `retry_recoveries`, `mean_attempts_per_reading`, and `failed_deliveries`. This demonstrates the reliability-versus-network-cost trade-off. `metrics.json` separately records detector recall and false-positive rate from training.

## Team responsibilities

| Member | Primary responsibility | Demo/report evidence |
| --- | --- | --- |
| Aditya | Network simulation and reliability experiments; maintain `simulator.py`. | Explain loss, attempts, delivery rate, retries, outages, and the four experiment reports. |
| Ashrit | Gateway and protocol; maintain `gateway.py`. | Run gateway, show `/summary`, explain packet schema, acknowledgement, deduplication, and alerts. |
| Tanvi | Dataset, model, edge scoring, and hardware bridge; maintain trainer/dataset documentation. | Explain CSV labels, `metrics.json`, verdict/reason bits, shared scoring logic, and bridge rationale. |

Everyone should be able to run the two-terminal demo. Suggested speaking order: Aditya configures the network run, Ashrit shows delivery and alerts at the gateway, then Tanvi explains detection quality and the hardware bridge.

### File ownership and generated outputs

Ownership means the named member is responsible for understanding, explaining, and maintaining the file; all members should still be able to run the complete demo.

| Member | Maintained source and documentation files | Generated files they should explain |
| --- | --- | --- |
| Aditya | `simulator.py` | `demo_report.json`, `simulation_test_report.json`, `simulation_validation_report.json`, and experiment reports such as `baseline.json` or `outage.json` |
| Ashrit | `gateway.py` | `gateway_events.jsonl`, `gateway_demo_validation.jsonl`, `gateway_test_events.jsonl`, and `gateway_validation_events.jsonl` |
| Tanvi | `build_dataset.c`, `train_model.c`, `scoring.h`, `jal_sakhi_node.ino`, `README.md`, and this guide | `dataset.csv`, `model_params.h`, and `metrics.json` |
| Shared build support | `build.ps1` and `Makefile`; Tanvi owns the data/model build steps and Aditya verifies that the complete simulator workflow runs. | `build_dataset.exe` and `train_model.exe` |

The following are **byproducts**, not files you would normally edit by hand:

| File pattern | Produced by | Purpose |
| --- | --- | --- |
| `*.exe` | `build.ps1` / `make` | Compiled C programs used to generate data and train the model. |
| `dataset.csv` | `build_dataset.exe` | Generated sensor-data input for the simulator. Replace only when using measured data. |
| `model_params.h` | `train_model.exe` | Generated model thresholds and parameters used by the simulator/reference firmware. |
| `metrics.json` | `train_model.exe` | Model-evaluation summary. |
| `*_report.json`, `demo_report.json` | `simulator.py` | Saved results of a particular network experiment. |
| `gateway*.jsonl` | `gateway.py` | Append-only log of packets accepted in a particular gateway run. |

`README.md`, `PROJECT_GUIDE.md`, all `.c`, `.h`, `.py`, `.ino`, `.ps1`, and `Makefile` files are project source/documentation and should be version-controlled. Generated outputs can be kept as demo evidence, but they may be safely regenerated by rerunning the relevant command.
