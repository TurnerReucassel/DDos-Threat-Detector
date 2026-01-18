Readme.md

PROBLEM to be detected: DDos threats by analyzing network traffic (using public IDS datasets)
Tech Stack:
Frontend: React + Chart.js
Backend: FastAPI
DB: PostgreSQL
Container: Docker (docker-compose)
Deploy later: AWS

aiming to finish: 
1. Train a baseline DDoS detector on a public IDS dataset with an anti-leak split
- Save: model + features.json + threshold.json + metrics.json
2. FastAPI runs with:
- GET /health
- POST /score (validate input → score → store)
- GET /alerts (dashboard query)
3. Postgres stores raw events + scores + alerts
4. React page that:
- calls GET /alerts
- displays a Chart.js visualization (even simple: “alerts over time”)
- A single docker compose up brings up api + db + frontend


right now: 
v1 task: binary detection (DDoS vs benign)
input format: flow feature row JSON (not PCAP parsing today)
target metric: PR-AUC + Recall at fixed FPR (ex: FPR ≤ 1%)
explicit non-goals today: transformers, Kafka, multi-class, malware, live capture
