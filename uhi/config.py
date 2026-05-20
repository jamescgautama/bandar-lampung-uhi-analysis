import os

# Earth Engine Configuration
GEE_PROJECT = 'urban-heat-island-482910'
ROI_PATH = 'export.geojson'
EXPORT_DESCRIPTION = 'uhi_data_multiyear_export'

# Data Collection Parameters
YEARS = [2019, 2022, 2025]
SAMPLE_SIZE = 10000
GRID_CELL_SIZE = 750
DRY_SEASON_START = '05-01'
DRY_SEASON_END = '09-30'

# ML Analysis Parameters
CSV_PATH = 'uhi_data_multiyear_export.csv'
TARGET_COL = 'LST'
FEATURE_COLS = ['MNDWI', 'NDBI', 'NDVI', 'dist_coast', 'elevation', 'slope', 'year']
FEATURES_TO_AGG = ['LST', 'MNDWI', 'NDBI', 'NDVI', 'dist_coast', 'elevation', 'slope']
GROUP_COL = 'grid_id'
N_SPLITS = 5
RANDOM_STATE = 42
