# TrainStation

**Tagline:** _All aboard for fast, guided ML experimentation._

TrainStation is an internal ML accelerator designed to make model experimentation fast, simple, and accessible for developers.

## Features
- AutoML support (FLAML, AutoKeras, Darts)
- Tasks: Image Classification, Object Detection, Time Series, Regression
- Streamlit UI for user-friendly interaction
- GPU-enabled Docker environment
- VS Code DevContainer for consistent development

## Getting Started
```bash
docker build -t trainstation .
docker run --gpus all -p 8501:8501 trainstation
```

Or open in VS Code and "Reopen in Container".
