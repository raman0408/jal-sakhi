# Jal Sakhi — Software Network Simulation

Jal Sakhi is a Computer Networks course project that simulates a village water-monitoring node sending readings to a gateway over an unreliable LoRa-like link. No physical hardware is required.

The ESP32 sketch is retained as a reference design showing how the same packet and scoring logic could run on hardware. The assessed runnable system is the CSV replay, network simulation, and gateway.

## What runs

- `build_dataset.c` creates a deterministic synthetic water-quality dataset when no measured CSV is available.
- `train_model.c` fits the Mahalanobis detector and exports `model_params.h`.
- `simulator.py` replays readings as node packets, applies loss/retries/outages, and posts successful packets to the gateway.
- `gateway.py` is a local HTTP gateway. It deduplicates packets and opens an alert ticket for unsafe readings.
- `scoring.h` is the shared C implementation intended for both the trainer and the reference firmware.

## Quick start on Windows

Requirements: Python 3.10+ and GCC (MinGW is sufficient). No Python packages are needed.

Open two PowerShell terminals in this folder.

Terminal 1:

```powershell
python gateway.py
```

Terminal 2:

```powershell
.\build.ps1
python simulator.py --loss 0.10 --retries 2
```

The build script generates `dataset.csv`, rebuilds `model_params.h`, and writes evaluation results to `metrics.json`. The simulation writes `simulation_report.json`; the gateway records delivered packets in `gateway_events.jsonl`.

To inspect the gateway while it is running, open `http://127.0.0.1:5000/summary` or `http://127.0.0.1:5000/health`.

## Network experiments

Use the same CSV/model while changing network conditions:

```powershell
# Replay the first 200 readings under 25% packet loss.
python simulator.py --limit 200 --loss 0.25 --retries 2

# Simulate a gateway/network outage for every 20th node transmission.
python simulator.py --limit 200 --loss 0.05 --outage-every 20
```

`simulation_report.json` reports readings replayed, transmission attempts, delivery rate, simulated packet drops, unavailable gateway attempts, watch readings, and unsafe readings. These are the primary Computer Networks evaluation measures; `metrics.json` separately records detector recall and false-positive rate.

## Using your own dataset

Provide a CSV with this header and run the trainer against it:

```text
ts,tds,turbidity,ph,temp_c,level_pct,label,source
```

`label` is `0` for normal and `1` for a contamination event. At least three normal rows and one event row are required.

```powershell
.\train_model.exe --data your_readings.csv --header model_params.h --seed 11
python simulator.py --data your_readings.csv
```

## Reference hardware design

`jal_sakhi_node.ino` is not required to run this project. It documents the proposed ESP32 sensor, filtering, alert-confirmation, and LoRa-packet design for the hardware extension. Its external Arduino libraries and physical sensors are intentionally out of scope for the simulation.

## Optional GNU Make

On Linux/macOS or a Windows environment with GNU Make:

```bash
make all-steps
```
