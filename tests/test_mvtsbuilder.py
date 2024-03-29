import mvtsbuilder
import pandas as pd

def test_mvtsbuilder():
    work_dir = '/Users/jiaxingqiu/Documents/CAMA_projects/BSI/code/projects/case_fwd'
    df = pd.read_csv("/Users/jiaxingqiu/Documents/CAMA_projects/BSI/code/projects/raw_data/bsi_new_deidentified_bc.csv", nrows=20000)
    prj = mvtsbuilder.Project(work_dir)
    prj.def_episode_from_json()
    prj.def_episode(input_time_len=2*24*60, output_time_len=2*24*60, time_resolution=60, time_lag=0, anchor_gap=7*24*60)
    prj.build_mvts(source=df, nsbj=41)
    prj.mvts_df.__uid.nunique()
