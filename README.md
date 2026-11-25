# Sweatpants Modules

Open-source automation modules for [Sweatpants](https://github.com/Extra-Chill/sweatpants).

## Modules

| Module | Description |
|--------|-------------|
| [agent-ping-webhook](agent-ping-webhook/) | Receive Data Machine Agent Ping webhooks and route them to AI agents via Kimaki CLI |
| [audio-transcription](audio-transcription/) | Transcribe audio files with speaker diarization using Whisper and PyAnnote |

| [mediavine-reports](mediavine-reports/) | Export per-URL Mediavine revenue reports and deliver them via signed callbacks |

## Installation

Modules are standalone — clone or copy the module directory into your Sweatpants modules path.

## Deployment

Merging to `main` automatically deploys changed modules to production. Only touched top-level directories that still contain `module.json` are installed; root-only changes and deleted modules are skipped. When the previous commit is unavailable, all modules are selected.

For a manual deployment, open **Actions → Deploy Sweatpants modules → Run workflow** on `main`. The `modules` input accepts `all` (default) or a comma- or whitespace-separated list, such as `audio-transcription, mediavine-reports`. Invalid names fail before deployment.

Configuration:
- `SWEATPANTS_API_AUTH_TOKEN`: required secret for this repository, held **only in the GitHub `production` environment** (Settings → Environments → production → Environment secrets). Use the master bearer token with `modules:admin` scope, not a limited token. Do not duplicate it as a repository-wide secret.
- `SWEATPANTS_URL`: optional repository variable; defaults to `https://sweatpants.chubes.net`.

The workflow installs each selected module using `POST /modules/install-git`, then calls `POST /modules/reload` once if any install succeeded. Failed installs do not prevent other modules from being attempted; failures are reported in the workflow summary and fail the run. It does **not** use `/modules/sync`: sync only installs entries from the server's own `modules.yaml`, so it cannot deploy newly added modules automatically.

Run the isolated detector tests locally with `bash .github/scripts/test-changed-modules.sh`.

## Contributing

Each module is a self-contained directory with:
- `module.json` — Sweatpants module manifest
- `main.py` — Module entrypoint (subclasses `sweatpants.Module`)
- `README.md` — Setup and usage docs
