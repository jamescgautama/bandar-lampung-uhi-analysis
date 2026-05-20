import ee
import os
from . import config
from . import utils

class DataCollector:
    def __init__(self):
        utils.initialize_ee(config.GEE_PROJECT)
        self.roi = utils.get_region_of_interest(config.ROI_PATH)

    def get_environmental_variables(self):
        """Calculates elevation, slope, and distance to coast."""
        elevation = ee.Image("NASA/NASADEM_HGT/001").select('elevation').clip(self.roi)
        slope = ee.Terrain.slope(elevation).rename('slope')
        
        # Distance to coast
        proj = ee.Projection("EPSG:32748").atScale(10)
        roi_context = self.roi.buffer(100000)
        worldcover = ee.Image("ESA/WorldCover/v200/2021").select("Map").clip(roi_context).reproject(proj)
        
        water = worldcover.eq(80)
        land = water.Not()
        coast = land.focal_min(1).neq(land).selfMask()
        dist_pixels = coast.fastDistanceTransform(256).sqrt()
        pixel_size = worldcover.projection().nominalScale()
        
        dist_coast = dist_pixels.multiply(pixel_size).rename("dist_coast").updateMask(land).clip(self.roi)
        
        return elevation, slope, dist_coast

    def process_year(self, year, sample_size=config.SAMPLE_SIZE):
        print(f"Processing year: {year}...")
        
        start_date = f'{year}-{config.DRY_SEASON_START}'
        end_date = f'{year}-{config.DRY_SEASON_END}'
        
        # --- 1. OPTICAL (Sentinel-2) ---
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterBounds(self.roi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))

        s2_cloud_prob = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY") \
            .filterBounds(self.roi) \
            .filterDate(start_date, end_date)

        s2_with_cloud_mask = ee.Join.saveFirst('cloud_probability').apply(
            primary=s2,
            secondary=s2_cloud_prob,
            condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
        )

        s2 = ee.ImageCollection(s2_with_cloud_mask) \
            .map(utils.mask_s2_clouds) \
            .median() \
            .clip(self.roi)
        
        s2_indices = utils.add_indices(s2)
        s2_final = s2_indices.select(['NDVI', 'NDBI', 'MNDWI'])

        # --- 2. THERMAL (Landsat 8/9) ---
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        if year >= 2022:
            l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            l8 = l8.merge(l9)
            
        l8_processed = l8.filterBounds(self.roi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUD_COVER', 50)) \
            .map(utils.mask_l8_clouds) \
            .map(utils.get_lst) \
            .median() \
            .clip(self.roi)
        
        l8_lst = l8_processed.select('LST')

        # --- 3. ENVIRONMENTAL VARIABLES ---
        elevation, slope, dist_coast = self.get_environmental_variables()

        # --- 4. STACK AND SAMPLE ---
        stack = s2_final.addBands([l8_lst, elevation, slope, dist_coast])
        stack = stack.set('year', year)
        
        year_img = ee.Image.constant(year).rename('year').clip(self.roi)
        stack_with_year = stack.addBands(year_img)
        
        samples = stack_with_year.sample(**{
            'region': self.roi,
            'scale': 10,
            'numPixels': sample_size,
            'geometries': True
        })
        
        # --- GRID SYSTEM ---
        grid = utils.make_grid(self.roi, cell_size=config.GRID_CELL_SIZE)
        grid = grid.map(lambda f: f.set('grid_id', f.id()))

        samples_with_grid = ee.Join.saveFirst('grid').apply(
            primary=samples,
            secondary=grid,
            condition=ee.Filter.intersects(leftField='.geo', rightField='.geo')
        )

        samples_final = samples_with_grid.map(
            lambda f: f.set('grid_id', ee.Feature(f.get('grid')).get('grid_id'))
        )

        return samples_final

    def run_export(self):
        years = config.YEARS
        collections = []
        
        for y in years:
            try:
                fc = self.process_year(y)
                fc_with_coords = fc.map(utils.add_coords)
                collections.append(fc_with_coords)
                print(f"Prepared collection for year {y}")
            except Exception as e:
                print(f"Error preparing year {y}: {e}")

        if collections:
            print("Merging collections...")
            merged_fc = ee.FeatureCollection(collections).flatten()
            
            task_config = {
                'collection': merged_fc,
                'description': config.EXPORT_DESCRIPTION,
                'fileFormat': 'CSV'
            }
            
            print("Starting export task to Drive...")
            task = ee.batch.Export.table.toDrive(**task_config)
            task.start()
            
            print(f"Task started: {task.id}")
        else:
            print("No collections to export.")
