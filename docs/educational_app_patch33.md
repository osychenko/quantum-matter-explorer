# Patch 33 — First local-first educational application

The backend wraps the strict Patch 32 curated contract. This is an **application prototype**, not a general QMC execution service or a certified many-body engine.

## Setup

From repository root, Python >=3.10:

```bash
python3 -m pip install -e '.[web]'
python3 -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

In a **second terminal**, from `web/`, with Node >=20:

```bash
npm install
npm run dev
```

Open **http://127.0.0.1:5173**. The Vite development server proxies `/api` to FastAPI port 8000. Do not publish either development server on a public interface. For a remote development machine use SSH forwarding:

```bash
ssh -L 5173:127.0.0.1:5173 -L 8000:127.0.0.1:8000 user@your-dev-host
```

The backend OpenAPI page is at http://127.0.0.1:8000/docs.

Run Python API tests with `python3 -m pytest -q tests/test_educational_api.py tests/test_educational_contract.py`, and frontend build with `cd web && npm run build` after `npm install`.

## Supported API

- `GET /api/v1/health`: status and contract version.
- `GET /api/v1/experiments`: only the two curated Patch 32 presets.
- `POST /api/v1/experiments/{preset_id}/run` JSON `{ "parameters": { "beta": 2.4, "cell_length": 2.8 } }` for `ideal-boson-addition-v1` or `{ "parameters": {} }` for `softcore-m8-baseline-recorded-v1`. All unknown parameters rejected by Patch 32.
- `POST /api/v1/experiments/ideal-boson-addition-v1/sweep`: JSON `{ "beta": 2.4, "cell_length": 2.8, "variable": "beta", "count": 25 }`; count 2..31. Deterministic analytic visualization points only.

No arbitrary sampler, research-only M=12 exchange, arbitrary filesystem location, or long-running job is callable over HTTP. No authentication or server-side notebook: **localhost only**. The client notebook is stored in browser `localStorage` (up to 20 saved results) and can be exported as JSON. No user accounts, remote tracking, or database.

## Scientific scope and next steps

The ideal-boson experiment calculates the canonical N=1->2 zero-mode midpoint imaginary-time correlation. Changing beta or box size affects the thermal partition sum. The M=8 soft-core benchmark is a **previously measured recording**, with its replica SEM and finite-basis spectral reference; it is not an interactive Monte Carlo computation or a continuum-extrapolated result. Interface language preserves the distinction.

Patch 34: educational 3D visuals, box/wavevector conceptual models, and interactive explanation. Patch 35: missions and progress based on prediction and interpretation, not on fabricated particle trajectories.
