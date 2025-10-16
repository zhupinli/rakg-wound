# -*- coding: utf-8 -*-
import argparse
from .pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="KG Cleaner compact pipeline")
    # you can add --config later if needed
    args = parser.parse_args()
    paths = run_pipeline()
    for k,v in paths.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()