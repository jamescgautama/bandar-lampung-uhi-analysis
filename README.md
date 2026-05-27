# Bandar Lampung Urban Heat Island (UHI) Analysis

An environmental analysis tool utilizing the Google Earth Engine (GEE) API and Machine Learning to model Urban Heat Island (UHI) intensity and identify thermodynamic drivers in Bandar Lampung, Indonesia.

## Features

- **Automated GEE Collection**: Retrieves and filters Landsat 8/9 and Sentinel-2 imagery for target years, exporting data gridded at 750m spatial resolution.
- **Surface Temperature Prediction**: Trains Scikit-Learn Random Forest and XGBoost regressors to predict Land Surface Temperature (LST).
- **SHAP Explanation**: Explains model predictions using SHAP (SHapley Additive exPlanations) values to identify the primary drivers of localized heating.
- **Geospatial Mapping**: Exports analysis regions as GeoJSON boundary structures (`export.geojson`) for frontend integration.

## Architecture Overview

```text
       +-------------------------------------------------------+
       |             Google Earth Engine (GEE) API             |
       |  (Landsat 8/9 LST & Sentinel-2 Vegetation Indices)   |
       +-------------------------------------------------------+
                                   |
                                   v  GEE Task Export (Google Drive)
       +-------------------------------------------------------+
       |             uhi_data_multiyear_export.csv             |
       +-------------------------------------------------------+
                                   |
                                   v  python main.py --analyze
       +-------------------------------------------------------+
       |             Pandas Preprocessing & Grid Align         |
       +-------------------------------------------------------+
                                   |
                   +---------------+---------------+
                   |                               |
                   v                               v
       [ Random Forest Regressor ]         [ XGBoost Regressor ]
                   |                               |
                   +---------------+---------------+
                                   |
                                   v
       +-------------------------------------------------------+
       |           SHAP Feature Importance & Plots             |
       +-------------------------------------------------------+
```

## Tech Stack

- **Geospatial Platform**: Google Earth Engine (Python API)
- **Language**: Python 3.12+
- **Machine Learning**: Scikit-Learn, XGBoost, SHAP
- **Data Wrangling**: Pandas, GeoPandas, NumPy

## Setup

### Prerequisites
- Python 3.12+
- Active Google Earth Engine account
- GEE project authenticated on your machine

### Installation & Auth
1. Set up a virtual environment and install packages:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # or
   venv\Scripts\activate     # On Windows
   
   pip install -r requirements.txt
   ```
2. Authenticate your Earth Engine environment:
   ```bash
   earthengine authenticate
   ```
3. Open `uhi/config.py` and update the GEE project settings:
   ```python
   GEE_PROJECT_ID = 'your-gee-project-id'
   ```

## Usage

All commands are controlled using the `main.py` entrypoint script.

### 1. Collect Data from Earth Engine
Trigger a task on GEE to process satellite images and export the tabular grid data to your Google Drive:
```bash
python main.py --collect
```
Once the export task finishes on your GEE console, download the output CSV file to the root of this project folder under the name `uhi_data_multiyear_export.csv`.

### 2. Run Modeling & Analysis
Preprocess the downloaded dataset, split training boundaries, fit machine learning regressors, and show SHAP impact values:
```bash
python main.py --analyze
```

## Environmental Indices & Analysis

The analysis predicts Land Surface Temperature (LST) using several remote sensing indicators:
- **NDVI (Normalized Difference Vegetation Index)**: Measures vegetation density and health (acts as a cooling factor).
- **NDBI (Normalized Difference Built-Up Index)**: Highlights man-made structures, buildings, and concrete surfaces (acts as a heating factor).
- **MNDWI (Modified Normalized Difference Water Index)**: Maps open water bodies.
- **Elevation / Coast Proximity**: Incorporates digital elevation models to evaluate terrain-based cooling and sea-breeze influences.

### Key Modeling Findings
- **Predictive Performance**: Random Forest achieves the highest performance with an $R^2 \approx 0.85$.
- **Thermodynamic Drivers**: Built-up densities (NDBI) are directly proportional to surface temperature spikes, while dense vegetation (NDVI) is the strongest cooling buffer.
- **Elevation and Coastal Cooling**: Proximity to the coast and higher elevation act as secondary cooling mitigators against urban heat retention.
