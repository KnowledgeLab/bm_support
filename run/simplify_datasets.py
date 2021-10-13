import os
import argparse
import pandas as pd

versions = [("gw", 11, 12), ("lit", 8, 12)]
current_version = 1


def main():
    for origin, version, subversion in versions:
        df_path = os.path.expanduser("~/data/kl/final/{0}_{1}_{2}.h5".format(origin, version, subversion))
        df = pd.read_hdf(df_path, key="df")
        columns_of_interest = ["up", "dn", "pmid", "pos", "cdf_exp"]
        df[columns_of_interest].to_csv(f"~/data/kl/external/{origin}_v{current_version}.csv.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()
