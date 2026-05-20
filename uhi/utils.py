import ee
import json
import os

def initialize_ee(project):
    """Initializes Earth Engine."""
    try:
        ee.Initialize(project=project)
    except Exception as e:
        print(f"Earth Engine authentication failed or not initialized: {e}")
        return False
    return True

def get_region_of_interest(geojson_path):
    """Parses GeoJSON to find the boundary."""
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")
        
    with open(geojson_path) as f:
        data = json.load(f)
    
    # Assuming the first feature is the boundary of interest
    feature = data['features'][0]
    geom = feature['geometry']
    ee_geom = ee.Geometry(geom)
    return ee_geom

def mask_s2_clouds(image):
    """Masks clouds in Sentinel-2 images using Cloud Probability."""
    cloud_prob = ee.Image(image.get('cloud_probability'))
    prob = cloud_prob.select('probability')
    is_cloud = prob.gt(50)
    return image.updateMask(is_cloud.Not()).divide(10000)

def add_indices(image):
    """Calculates and adds NDBI, NDVI, MNDWI bands."""
    ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI') 
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI')
    
    return image.addBands([ndbi, ndvi, mndwi])

def get_lst(image):
    """Calculates Land Surface Temperature (LST) from Landsat 8/9."""
    # Scale factors for Collection 2 Surface Temperature Band: ST_B10
    st = image.select('ST_B10')
    
    # Mask invalid values (0 is fill)
    mask = st.gt(0)
    st = st.updateMask(mask)
    
    # Rescale to Kelvin and convert to Celsius
    st_kelvin = st.multiply(0.00341802).add(149.0)
    lst_celsius = st_kelvin.subtract(273.15).rename('LST')
    
    return image.addBands(lst_celsius)

def mask_l8_clouds(image):
    """Masks clouds and fill in Landsat 8 images using QA_PIXEL."""
    qa = image.select('QA_PIXEL')
    
    # Bit 0: Fill, Bit 3: Cloud, Bit 4: Cloud Shadow
    mask = qa.bitwiseAnd(1 << 0).eq(0) \
        .And(qa.bitwiseAnd(1 << 3).eq(0)) \
        .And(qa.bitwiseAnd(1 << 4).eq(0))
    
    return image.updateMask(mask)

def make_grid(roi, cell_size, projection="EPSG:32748"):
    """Creates a covering grid over the ROI."""
    proj = ee.Projection(projection)
    grid = roi.transform(proj, 1).coveringGrid(proj.atScale(cell_size))
    return grid

def add_coords(feat):
    """Adds longitude and latitude to feature properties."""
    coords = feat.geometry().coordinates()
    return feat.set({
        'longitude': coords.get(0),
        'latitude': coords.get(1)
    })
