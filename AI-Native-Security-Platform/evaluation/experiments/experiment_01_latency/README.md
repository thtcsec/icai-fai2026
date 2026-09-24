# Experiment 01 (legacy folder)

**Deprecated.** The paper latency table is produced by:

```bash
# Redis 7.x on localhost:6379 required
python ../../results/generate_table2.py
```

That path measures: INT8 inference → identity fusion → Redis `XADD`+`XREADGROUP` → DQN → SOAR playbook **construction** (no OpenFlow install).

Do not use the scripts in this folder for artifact evaluation; they are historical stubs.
