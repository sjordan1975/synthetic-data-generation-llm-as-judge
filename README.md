# AI Synthetic Data Generator

Minimal setup to get this project running with a project-specific virtual environment.

## Prerequisites

- Python 3.13+
- `pip`

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create local env values from the example file:

```bash
cp .env.example .env.local
```

Then edit `.env.local` with your real keys / settings.

## Verify Setup

```bash
python -V
pip -V
pip list
```