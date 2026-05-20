# Bandar Lampung Urban Heat Island (UHI) Analysis

A tool to analyze Urban Heat Island effects using Google Earth Engine and Machine Learning. It looks at Bandar Lampung, Indonesia, to see how things like vegetation and buildings affect surface temperature.

## Setup

### Requirements
* Google Earth Engine account
* Python 3.12+
* Authenticate GEE by running `earthengine authenticate`

### Install
1. Clone this repo and go into the folder.
2. Set up a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Update `uhi/config.py` with your Earth Engine Project ID.

## Usage

Use `main.py` for everything.

### 1. Collect Data
Starts a GEE task to export data to your Google Drive.
```bash
python main.py --collect
```
Once it's done, download the CSV to this folder.

### 2. Run Analysis
Preprocesses data and trains models (Random Forest, XGBoost, etc.).
```bash
python main.py --analyze
```

## Summary

### Problem
Bandar Lampung is getting hotter as it builds up. This project finds out why by looking at temperature (LST) versus vegetation (NDVI), buildings (NDBI), and water (MNDWI).

### How it works
1. Pulls data from Sentinel-2 and Landsat 8/9 via GEE.
2. Grids everything to 750m resolution.
3. Uses Random Forest and XGBoost to predict temperature.
4. Uses SHAP to see which factors matter most.

### Results
* Random Forest works best (R² ~0.85).
* More buildings (NDBI) = more heat.
* More trees (NDVI) = less heat.
* Elevation and being near the coast also help cool things down.
