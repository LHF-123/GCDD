# WebFG-496 deployment bundle

This bundle is a source-only snapshot of the `major-revision` branch. It does
not contain WebFG images, DINOv2 weights/cache, checkpoints, generated
outputs, or any other historical experiment artifact.

## New-server setup

1. Copy the ZIP package to the server and extract it. The archive contains a
   self-contained repository source tree; it does not need a Git checkout to
   execute.
2. Activate an existing compatible Python/CUDA environment and install the
   project dependencies only in that environment if they are not already
   present. Do not modify an environment used by another running experiment.
3. Provide a WebFG root whose direct children are `web-bird`, `web-car`, and
   `web-aircraft`. Each child must retain the official `train/`, `val/`,
   `train-list.txt`, and `val-list.txt` structure. The `.tar.gz` files are
   optional: if present, the seven known corrupt members are inspected; if no
   valid original member is available, the fixed WebFG-only exclusion manifest
   is regenerated.
4. If DINOv2 is already cached offline, set `TORCH_HOME` for this shell to its
   cache location. This setting is process-local and does not change other
   jobs.

## Commands

Use the actual data path explicitly; it is never hard-coded into the launcher:

```bash
cd gcdd-webfg496
PYTHON_BIN=python bash scripts/webfg496/run_all.sh --preflight --data-root /data/webfg496
PYTHON_BIN=python bash scripts/webfg496/run_all.sh --resume --data-root /data/webfg496
```

The second command performs the 36-run matrix only after the first command's
checks pass and the selected GPU is exclusively idle. It runs one job at a time
on the configured GPU. To detach from SSH:

```bash
PYTHON_BIN=python bash scripts/webfg496/run_all.sh --background --data-root /data/webfg496
```

Use the same optional `--data-root` and, if desired, a dedicated
`--output-root /data/webfg-results` with read-only operations:

```bash
bash scripts/webfg496/run_all.sh --status --output-root /data/webfg-results
bash scripts/webfg496/run_all.sh --summary --output-root /data/webfg-results
```

Without `--output-root`, results are isolated under
`outputs_major_revision/webfg496/` inside the extracted source tree.
