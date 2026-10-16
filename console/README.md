# PolicyJEV console

A local moderation console for the PolicyJEV case study. It has three pages:

- **Score**: type a comment and see what logistic regression, DistilBERT and JEV decide for each of the six labels, and what the moderation policy does with it.
- **Policy**: an explorer for the LOW and HIGH thresholds, then the test-set policy results (where the comments went, five measures, JEV against each baseline, and what validation promised against what the test set delivered).
- **Results**: average precision by label and recall of the rare labels against the review budget.

Everything runs on your machine. The API binds to 127.0.0.1, and nothing you type is stored or logged. Every number on the Policy and Results pages comes from `GET /api/results`, which reads the committed metric files; nothing is recomputed.

## Install

From the repository root, with Python 3.13 and a virtual environment active (see the root `README.md`), install the Python dependencies and the package:

```
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` already pins the API packages (FastAPI, uvicorn, httpx), so the `.[api]` extra in `pyproject.toml` is not needed after it.

Then install the front end:

```
cd console
npm install
```

The two fonts (Bricolage Grotesque and Instrument Sans) are installed from npm packages and served with the app. The app makes no request to a font service or any other outside address.

## Run

Use two terminals.

Terminal 1, from the repository root:

```
python -m policyjev.api
```

The API starts on http://127.0.0.1:8000 and loads the local models in the background. The first Score request can take several seconds until the top bar stops showing the models as loading.

Terminal 2, from `console/`:

```
npm run dev
```

Open http://127.0.0.1:5173. The dev server proxies `/api` to the API. Restart the API after pulling changes: the Policy page needs the `policy_per_label` section that newer versions of `/api/results` return.

## What is sent where

- **Score as I type** (on by default, 320 ms pause) scores only logistic regression and DistilBERT, which run on this machine.
- **JEV** runs only when you press Score or Ctrl+Enter. It needs a TypeSafe API key in the `TYPESAFE_API_KEY` environment variable of the API server (see `.env.example` in the repository root). Without it the top bar shows JEV as needing a key and the other two systems still score.
- JEV requests are sent to the TypeSafe API. They are not cached, but the text leaves your machine.

## Scripts

- `npm run dev`: development server on port 5173.
- `npm run build`: type check and production build into `dist/`.
- `npm run typecheck`: type check only.
- `npm test`: unit and component tests.
- `npm run shots`: after `npm run build` and with the API running, serves the built app (proxying `/api` to the API, `API_URL` if it is not on port 8000), saves screenshots at 1440, 1024, 768 and 375 px to `.shots/`, and checks that no page scrolls sideways from 360 px up. Only benign text is typed. Needs `npx playwright install chromium` once.
- `node scripts/fontcheck.mjs`: checks in Chromium that both font families load, that the weight axis works and that tabular figures give equal-width digits.

## Keeping the decision rule in step with the policy engine

The Score and Policy pages decide in TypeScript (`src/decision.ts`). `tests/test_decision_parity.py` checks the same rule against `src/policyjev/policy/engine.py` for every system and label of the real `/api/thresholds` response and writes `src/fixtures/decision_parity.json`, which the front-end tests replay. If the committed thresholds change, regenerate the file from the repository root:

```
python tests/test_decision_parity.py
```
