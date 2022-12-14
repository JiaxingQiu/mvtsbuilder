
import pandas as pd
from mvtsbuilder.functions.make_sample_info_from_csv import *
from mvtsbuilder.functions.make_sbjs_ts import *
from mvtsbuilder.functions.make_episodes_ts import *
from mvtsbuilder.functions.viz_episode_ts import *

def make_mvts_df_from_csv(df_csv_fullname_ls, nsbj, frac, source_dict, variable_dict, input_time_len, output_time_len, time_resolution, time_lag, anchor_gap, topn_eps=None, stratify_by=None, viz=False, viz_ts=False, dummy_na=False, return_episode=True, skip_uid=None, keep_uid=None, df_raw=None):
    
    if df_raw is None:
        df_sample_info, df_csv_fullname_ls_updated, msg = make_sample_info_from_csv(df_file_dict=df_csv_fullname_ls, source_dict=source_dict, variable_dict=variable_dict, nsbj=nsbj, frac=frac, stratify_by=stratify_by, skip_uid=skip_uid, keep_uid=keep_uid)
    else:
        df_sample_info = pd.DataFrame({"filename":["NA"], "fullname":["NA"], "source_key":["external"], "file_key":["df_raw"], "id":["NA"], "__uid":["NA"], "already_sampled":[0]})
        df_csv_fullname_ls_updated = pd.DataFrame({"filename":["NA"], "fullname":["NA"], "source_key":["external"], "file_key":["df_raw"], "id":["NA"], "__uid":["NA"], "already_sampled":[1]})
        msg = "using external df_raw instead!"

    df_sbjs_ts = make_sbjs_ts(df_sample_info, variable_dict, time_resolution, viz=viz, dummy_na=dummy_na, df_raw=df_raw) # patient subset dataframe table
         
    if viz:
        fig_title = "Visualize Cleaned DataFrame"
        viz_df_hist(df_sbjs_ts, fig_title)
    
    df_episodes_ts = None
    if return_episode:
        df_episodes_ts = make_episodes_ts(df_sbjs_ts, variable_dict, input_time_len, output_time_len, time_resolution, time_lag, anchor_gap, topn_eps=topn_eps)
    
    if viz:
        fig_title = "Visualize Episode DataFrame"
        viz_df_hist(df_episodes_ts, fig_title)
    if viz_ts:
        fig_title = "Visualize Episode Data Variables over Time"
        viz_episode_ts(df_episodes_ts, fig_title)
    return df_episodes_ts, df_csv_fullname_ls_updated, msg, df_sbjs_ts
