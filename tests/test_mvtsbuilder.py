import mvtsbuilder
import pandas as pd
import time

def test_mvtsbuilder():
    work_dir = '/Users/jiaxingqiu/Documents/CAMA_projects/BSI/case_uva'
    df = pd.read_csv("/Users/jiaxingqiu/Documents/CAMA_projects/BSI/case_uva/raw_data/bsi_new_deidentified_bc.csv", nrows=100000)
    prj = mvtsbuilder.Project(work_dir)
    prj.def_episode_from_json()
    prj.def_episode(input_time_len=2*24*60, output_time_len=2*24*60, time_resolution=60, time_lag=0, anchor_gap=7*24*60)
    tic = time.perf_counter()
    prj.build_mvts(source=df, frac=1.0)
    toc = time.perf_counter()
    print(f"Elapsed: {toc - tic:0.4f} seconds")
    prj.mvts_df.__uid.nunique()
