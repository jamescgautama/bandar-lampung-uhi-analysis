import argparse
import os
from uhi.data_collector import DataCollector
from uhi.modeler import UHIModeler
from uhi import config

def main():
    parser = argparse.ArgumentParser(description="Urban Heat Island Analysis Tool")
    parser.add_argument("--collect", action="store_true", help="Run Earth Engine data collection")
    parser.add_argument("--analyze", action="store_true", help="Run ML analysis")
    parser.add_argument("--all", action="store_true", help="Run both collection and analysis")
    
    args = parser.parse_args()

    if args.all or (not args.collect and not args.analyze):
        # Default behavior if no flags or --all
        run_collect = True
        run_analyze = True
    else:
        run_collect = args.collect
        run_analyze = args.analyze

    if run_collect:
        print("--- Starting Data Collection ---")
        collector = DataCollector()
        collector.run_export()
        print("Data collection task started. Please wait for the GEE task to complete and download the CSV.")

    if run_analyze:
        print("--- Starting ML Analysis ---")
        if os.path.exists(config.CSV_PATH):
            modeler = UHIModeler()
            modeler.run_stability_check()
        else:
            print(f"Error: {config.CSV_PATH} not found. Run with --collect first.")

if __name__ == "__main__":
    main()
