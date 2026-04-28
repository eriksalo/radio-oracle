# Workstream 5: Updates & Reliability

Make the Oracle boot reliably, stay running, recover from failures, and accept updates.

## Scope

- systemd service hardening
- Health checks and watchdog
- Auto-recovery from crashes
- OTA update mechanism (git pull + restart)
- Logging and diagnostics
- Backup and restore

## Key Files

```
systemd/
  oracle.service           # systemd unit file
oracle/
  health.py                # Subsystem health checks
scripts/
  setup_jetson.sh          # One-time Jetson setup
  migrate_rootfs.sh        # SD-to-NVMe migration
```

## Interface Contract

**Reads from all workstreams**: `health.py` checks Ollama, ChromaDB, audio devices, GPIO, disk space, etc.

**No code changes required in other workstreams** — reliability wraps around the existing system.

## TODO

- [ ] systemd watchdog integration (sd_notify)
- [ ] `oracle health` CLI subcommand (check all subsystems, report status)
- [ ] Structured logging (JSON) for log aggregation
- [ ] Auto-restart backoff (avoid crash loops)
- [ ] OTA update script: git pull, pip install, systemctl restart
- [ ] Disk space monitoring (ChromaDB + music can fill 1TB)
- [ ] Nightly SQLite vacuum / WAL checkpoint
- [ ] Read-only rootfs with writable overlay for data
