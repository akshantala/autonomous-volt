# Autonomous Volt · Pre-Autonomous Checklist + LUT (Web App)

Streamlit web app that runs the pre-autonomous safety checklist, logs runs to
Excel, predicts torque using a 3-D lookup table built from real Chevy Bolt
test data, and runs a closed-loop simulation comparing the baseline PID
controller against the LUT-based controller.

## Features — two tabs

**1. Pre-Autonomous Checklist** — checklist gating, post-run logging,
LUT torque prediction, and a re-query panel for exploring the LUT at
arbitrary conditions.

**2. Controller Comparison** — runs the same 30-second simulation as
Liam's Jupyter notebook directly in the browser. Click **Run Simulation**
to compare the baseline PID controller against the LUT-based feedforward
controller. Shows improvement summary cards, per-controller metrics
tables, and time-series plots of speed, torque, and jerk.

A toggle lets the user choose between **synthetic data** (matches the
simulation's idealized physics, giving a fair apples-to-apples controller
comparison) and **real CSV data** (your team's actual test recordings).
Use synthetic for the controller-improvement claim. Use real data for the
torque-prediction demo on the checklist tab.

## What's in this folder

```
volt_web_final/
├── app.py                  # Streamlit app (two tabs)
├── lut_module.py           # LUT physics + CSV loader + simulation engine
├── requirements.txt        # pip dependencies
├── .streamlit/
│   └── config.toml         # dark theme
├── data/                   # 19 real CSV test files
└── logs/                   # Excel logs written here
```

## Running locally

```bash
cd volt_web_final
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501.

## Deploying to a shareable URL (Streamlit Cloud)

1. Create a free GitHub account at [github.com](https://github.com)
2. Create a new public repo (e.g. `autonomous-volt`)
3. Upload all files in this folder, including the `data/` folder.
   On the empty repo page, click **"uploading an existing file"**, drag
   everything in (GitHub accepts up to 100 files per upload — drop the
   `data/` folder separately if needed), then click **Commit changes**.
4. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with
   GitHub → **Create app → Deploy from GitHub**
5. Repo: your repo · Branch: `main` · Main file: `app.py` → **Deploy**

Takes 2–3 minutes on first deploy. You'll get a URL like
`https://autonomous-volt-<username>.streamlit.app` to share.

## Presentation notes

- **For the controller comparison demo**, use the synthetic toggle.
  Liam's notebook numbers (+16 pp in-band time, +80% RMSE reduction,
  −36% torque slew) come from this comparison. The simulation is using
  an idealized bicycle model, so synthetic training data (generated from
  that same model) gives a fair comparison.
- **For the torque prediction demo**, use the real CSV data (default).
  The diagnostic expander shows the LUT was trained on 30,586 real samples
  from your team's tests.
- **The two demos answer different questions.** The comparison answers
  "is the LUT-based controller better than the PID-only baseline?" The
  prediction answers "what torque would the controller command for this
  run's conditions?"
