""" Project class. 2nd step in FAIR Medical AI framework, go through fucntionalities in data Projecting, following FAIR principle and 
    connecting to FAIRSCAPE web server.

    Module description details

    
    Usage example:

"""
from mvtsbuilder.functions.init_csv_fullname_ls import *
from mvtsbuilder.functions.make_mvts_df_from_csv import *
from mvtsbuilder.functions.cohort_fillna import *
from mvtsbuilder.functions.make_mvts_tfds_from_df import *
from mvtsbuilder.functions.extract_xy import *
from mvtsbuilder.classes.episode import Episode
import random
from datetime import datetime
import os
import json

class Project:
    
    def __init__(self, work_dir):
        self.hist = None # over all history
        self.datetime = None
        self.work_dir = None
        self.meta_dir = None
        self.variable_dict = None
        self.csv_source_dict = None
        self.sql_source_dict = None
        self.input_vars = None
        self.output_vars = None
        self.input_vars_byside = None
        self.output_vars_byside = None
        self.hist_def_episode = None# update by calling def_episode()
        self.episode = None 
        self.hist_build_mvts = None# update by calling build_mvts()
        self.sample_info = None 
        self.df_csv_fullname_ls = None
        self.sbj_df = None
        self.mvts_df = None
        self.hist_split_mvts = None # update by calling split_mvts()
        self.train_df = None 
        self.valid_df = None
        self.test_df = None
        self.train_df_imputed = None
        self.valid_df_imputed = None
        self.test_df_imputed = None
        self.train_tfds = None
        self.valid_tfds = None
        self.test_tfds = None
       

        # set initial datetime for a project instance
        self.datetime = str(datetime.now())
        print('Project begins at: ' + str(self.datetime))

        # set study working directory
        self.work_dir = work_dir
        os.chdir(self.work_dir)
        print('Working directory: ' + str(self.work_dir))

        # set meta data of dictionarys directory 
        self.meta_dir = str(self.work_dir) + '/meta_data'
        if not os.path.exists(self.meta_dir):
            os.mkdir(self.meta_dir)
        print('Meta_data directory: ' + str(self.meta_dir))

        # set variable_dict
        try:
            fullname = str(self.meta_dir)+'/variable_dict.json'
            f = open(fullname, "r")
            self.variable_dict = json.loads(f.read())
            print("Project variable dictionary loaded;")
        except:
            print("Unable to read variable dictionary. Please provide 'YOUR_PROJECT_WORKING_DIR/metadata/variable_dict.json' file.")
        # set csv_source_dict
        try:
            fullname = str(self.meta_dir)+'/csv_source_dict.json'
            f = open(fullname, "r")
            self.csv_source_dict = json.loads(f.read())
            print("Project csv_source dictionary loaded;")
        except:
            print("Unable to read csv source dictionary. Please provide 'YOUR_PROJECT_WORKING_DIR/metadata/csv_source_dict.json' file.")
        # set sql_source_dict
        try:
            fullname = str(self.meta_dir)+'/sql_source_dict.json'
            f = open(fullname, "r")
            self.sql_source_dict = json.loads(f.read())
            print("Project sql_source dictionary loaded;")
        except:
            print("Unable to read sql source dictionary. Please provide 'YOUR_PROJECT_WORKING_DIR/metadata/sql_source_dict.json' file.")
        
        # set input_vars/output_vars by variable_dict
        output_vars = []
        output_vars_byside = [] # variables that have "output:false", they are Projected but not included in final ML matrices
        for var_dict in self.variable_dict.keys():
            if 'output' in self.variable_dict[var_dict].keys():
                if self.variable_dict[var_dict]['output']:
                    if 'factor' in self.variable_dict[var_dict].keys():
                        l_list = list(self.variable_dict[var_dict]['factor']['levels'].keys())
                        v_list = [str(var_dict) + '___' + l for l in l_list]
                        output_vars = output_vars + v_list
                    if 'numeric' in self.variable_dict[var_dict].keys():
                        output_vars = output_vars + [var_dict]
                else:
                    if 'factor' in self.variable_dict[var_dict].keys():
                        l_list = list(self.variable_dict[var_dict]['factor']['levels'].keys())
                        v_list = [str(var_dict) + '___' + l for l in l_list]
                        output_vars_byside = output_vars_byside + v_list
                    if 'numeric' in self.variable_dict[var_dict].keys():
                        output_vars_byside = output_vars_byside + [var_dict]
        assert len(output_vars)>0, 'Project couldn\'t find output columns corresponding to the variable dictionary'
        input_vars = []
        input_vars_byside = [] # variables that have "input:false", they are Projected but not included in final ML matrices
        for var_dict in self.variable_dict.keys():
            if 'input' in self.variable_dict[var_dict].keys():
                if self.variable_dict[var_dict]['input']:
                    if 'factor' in self.variable_dict[var_dict].keys():
                        l_list = list(self.variable_dict[var_dict]['factor']['levels'].keys())
                        v_list = [str(var_dict) + '___' + l for l in l_list]
                        input_vars = input_vars + v_list
                    if 'numeric' in self.variable_dict[var_dict].keys():
                        input_vars = input_vars + [var_dict]
                else:
                    if 'factor' in self.variable_dict[var_dict].keys():
                        l_list = list(self.variable_dict[var_dict]['factor']['levels'].keys())
                        v_list = [str(var_dict) + '___' + l for l in l_list]
                        input_vars_byside = input_vars_byside + v_list
                    if 'numeric' in self.variable_dict[var_dict].keys():
                        input_vars_byside = input_vars_byside + [var_dict]
        assert len(input_vars)>0, 'Project couldn\'t find input columns corresponding to the variable dictionary'
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.input_vars_byside = input_vars_byside
        self.output_vars_byside = output_vars_byside
        
    def __str__(self):
        """print project updated constants and variables at any time

        Returns:
        """
        
        self.hist = {
            'Project Info':{
                'datetime': str(self.datetime),
                'working_dir': str(self.work_dir),
                'meta_data_dir': str(self.meta_dir),
                'ml_var':{
                    'inputs': str(self.input_vars),
                    'outputs': str(self.output_vars),
                    'others': str(self.input_vars_byside + self.output_vars_byside)
                }
            },
            'Episode Definition': self.hist_def_episode,
            'MVTS': self.hist_build_mvts,
            'MVTS for ML': self.hist_split_mvts
        }
        print(json.dumps(self.hist, indent=2))
        return 'Project Status'

    

    def def_episode(self, input_time_len, output_time_len, time_resolution=None, time_lag=None, anchor_gap=None):
        self.episode = Episode(input_time_len, output_time_len, self.variable_dict['__time']['unit'], time_resolution=time_resolution, time_lag=time_lag, anchor_gap=anchor_gap)
        # collect success history dictionary
        print("Success! Project has updated attributes --- episode. ")
        self.hist_def_episode = {'datetime': str(datetime.now())}
        self.hist_def_episode.update(self.episode.__dict__)

         
    def build_mvts(self, csv_pool_dir=None, df_raw=None, nsbj=None, frac=0.3, replace=True, topn_eps=None, viz=False, viz_ts=False, stratify_by=None, dummy_na=False, sep="---", return_episode=True, skip_uid=None, keep_uid=None):
       
        # make sure Project has episode object at hand by now
        if self.episode is None:
            return 'No episode defined -- you can use def_episode() to define one'
        episode = self.episode
        # validate argument csv_pool_dir and df_raw
        hist1 = "Not Run" # sampling
        if csv_pool_dir is None:
            print("Argument is None -- csv_pool_dir")
            if df_raw is None:
                print("Argument is None -- df_raw")
                return 'Neither csv_pool direstory nor external table object found -- please specify at least one'
            else: 
                print("Project is Projecting customized table format data ---")
                hist1 = {'from': 'given dataframe object'}
        else:
            print("csv_pool_dir: "+str(csv_pool_dir))
            if df_raw is None:
                # sampling from csv_pool
                if self.df_csv_fullname_ls is None:
                    self.df_csv_fullname_ls = init_csv_fullname_ls(csv_pool_dir, sep=sep)
                if replace:
                    print("Project is sampling with replacement from csv_pool --- ")
                    self.df_csv_fullname_ls = init_csv_fullname_ls(csv_pool_dir, sep=sep)
                    hist1 = {'from': 'csv_pool', 'path': str(csv_pool_dir), 'sep': str(sep)}
                else:
                    print("Project is sampling without replacement from csv_pool --- ")
                    hist1 = {'from': 'csv_pool', 'path': str(csv_pool_dir), 'sep': str(sep)}
            else:
                print("Project is Projecting customized table format data ---")
                hist1 = {'from': 'given dataframe object'}
        # special arguments
        hist2 = {
            'with_replacement': 'yes' if replace else 'no',
            'skip_uid': str(skip_uid) if skip_uid is not None else 'Not specified',
            'keep_uid': str(keep_uid) if keep_uid is not None else 'Not specified',
            'dummy_na': str(dummy_na),
            'stratify_by': str(stratify_by) if stratify_by is not None else 'None'
        }

        # building process
        hist3 = "Not Run" # 'return'
        self.mvts_df, self.df_csv_fullname_ls, self.sample_info, self.sbj_df = make_mvts_df_from_csv(self.df_csv_fullname_ls, nsbj, frac, self.csv_source_dict, self.variable_dict, episode.input_time_len, episode.output_time_len, episode.time_resolution, episode.time_lag, episode.anchor_gap, stratify_by=stratify_by, viz=viz, viz_ts=viz_ts, dummy_na=dummy_na, topn_eps=topn_eps, return_episode=return_episode, skip_uid=skip_uid, keep_uid=keep_uid, df_raw=df_raw)
        hist3 = {
            'nsubjects': len(self.mvts_df['__uid'].unique()) if self.mvts_df is not None else 'Unknown',
            'notes': str(self.sample_info),
            'episode_wise_df': 'YOUR PROJECT.mvts_df' if self.mvts_df is not None else 'No return',
            'subject_wise_df': 'YOUR PROJECT.sbj_df' if self.sbj_df is not None else 'No return'
        }
        # check input/output variable names
        if self.mvts_df is not None:
            output_vars = []
            output_vars_byside = [] # variables that have "output:false", they are Projected but not included in final ML matrices
            for var_dict in self.variable_dict.keys():
                if 'output' in self.variable_dict[var_dict].keys():
                    if self.variable_dict[var_dict]['output']:
                        output_vars = output_vars + list(self.mvts_df.columns[self.mvts_df.columns.str.startswith(str(var_dict))])
                    else:
                        output_vars_byside = output_vars_byside + list(self.mvts_df.columns[self.mvts_df.columns.str.startswith(str(var_dict))])
            assert len(output_vars)>0, 'Project couldn\'t find output columns corresponding to the variable dictionary'
            input_vars = []
            input_vars_byside = [] # variables that have "input:false", they are Projected but not included in final ML matrices
            for var_dict in self.variable_dict.keys():
                if 'input' in self.variable_dict[var_dict].keys():
                    if self.variable_dict[var_dict]['input']:
                        input_vars = input_vars + list(self.mvts_df.columns[self.mvts_df.columns.str.startswith(str(var_dict))])
                    else:
                        input_vars_byside = input_vars_byside + list(self.mvts_df.columns[self.mvts_df.columns.str.startswith(str(var_dict))])
            assert len(input_vars)>0, 'Project couldn\'t find input columns corresponding to the variable dictionary'
            # check input vars are the same before and after Project
            assert len(set(self.input_vars) & set(input_vars)) == len(self.input_vars), 'length of input_vars changed after Projecting'
            assert len(set(self.output_vars) & set(output_vars)) == len(self.output_vars), 'length of output_vars changed after Projecting'
            assert len(set(self.input_vars_byside) & set(input_vars_byside)) == len(self.input_vars_byside), 'length of input_vars_byside changed after Projecting'
            assert len(set(self.output_vars_byside) & set(output_vars_byside)) == len(self.output_vars_byside), 'length of output_vars_byside changed after Projecting'
            
            # reorder columns 
            base_vars = list(self.mvts_df.columns[~self.mvts_df.columns.isin(self.input_vars+self.output_vars+self.input_vars_byside+self.output_vars_byside)])
            self.mvts_df = self.mvts_df[self.input_vars+self.output_vars+self.input_vars_byside+self.output_vars_byside+base_vars] # mvts_df is not imputed
            
            #self.mvts_df_byside = self.mvts_df[self.input_vars_byside+self.output_vars_byside+base_vars]
            # add input output info to mvts_df object (attach new attributes to a pandas.DataFrame)
            try:
                self.mvts_df.attrs['input_vars'] = input_vars
                self.mvts_df.attrs['output_vars'] = output_vars
                self.mvts_df.attrs['input_vars_byside'] = input_vars_byside
                self.mvts_df.attrs['output_vars_byside'] = output_vars_byside
            except:
                print("add attrs to mvts_df failed")
                pass

            print("Success! Project has updated attributes --- mvts_df, sbj_df, sample_info, df_csv_fullname_ls")
            self.hist_build_mvts = {'datetime':str(datetime.now())}
            self.hist_build_mvts.update({'data_source':hist1})
            self.hist_build_mvts.update({'sampling':hist2})
            self.hist_build_mvts.update({'return':hist3})
            


    def split_mvts(self, valid_frac=0, test_frac=0, byepisode=False, batch_size=32, impute_input=None, impute_output=None, fill_value=-333, viz=False):
        
        # make sure Project has mvts_df object at hand by now
        if self.mvts_df is None:
            return 'No episode-wise MVTS dataframe defined -- you can use build_mvts() to build one'
        
        ######################### split ML train/valid/test data by fraction #########################
        # collect history for split
        hist1 = "Not Run"
        if byepisode:
            self.mvts_df['split_id'] = self.mvts_df['__uid'].astype(str) + self.mvts_df['__ep_order'].astype(str)
            hist1 = {'by': 'episode'}
        else:
            self.mvts_df['split_id'] = self.mvts_df['__uid']
            hist1 = {'by': 'subject'}
        all_list = list(self.mvts_df['split_id'].unique())
        random.shuffle(all_list)
        assert valid_frac>=0 and valid_frac<1, "validation set fraction must be >=0 and <1"
        assert test_frac>=0 and test_frac<1, "test set fraction must be >=0 and <1"
        train_frac = 1-valid_frac-test_frac
        assert train_frac>0 and train_frac<=1, "train set fraction must be >0 and <=1, please input legit valid_frac and/or test_frac"
        train_list = all_list[0:int(np.round(train_frac*len(all_list)))]
        self.train_df = self.mvts_df[self.mvts_df['split_id'].isin(train_list)]
        hist1.update({'train_size':len(list(self.train_df['split_id'].unique()))})
        self.train_df = self.train_df.drop(columns=['split_id'])
        if valid_frac > 0:
            if int(np.round(train_frac*len(all_list))) < len(all_list):
                valid_list = all_list[int(np.round(train_frac*len(all_list))):int(np.round((train_frac+valid_frac)*len(all_list)))]
                self.valid_df = self.mvts_df[self.mvts_df['split_id'].isin(valid_list)]
                hist1.update({'valid_size':len(list(self.valid_df['split_id'].unique()))})
                self.valid_df = self.valid_df.drop(columns=['split_id'])
        if test_frac > 0:
            if int(np.round((train_frac+valid_frac)*len(all_list))) < len(all_list): 
                test_list = all_list[int((train_frac+valid_frac)*len(all_list)):int(len(all_list))]
                self.test_df = self.mvts_df[self.mvts_df['split_id'].isin(test_list)]
                hist1.update({'test_size':len(list(self.test_df['split_id'].unique()))})
                self.test_df = self.test_df.drop(columns=['split_id'])
        print("Success! Project has updated attributes --- train_df, valid_df and test_df. ")
        hist1.update({'df_shape': {
            'train': str(self.train_df.shape),
            'valid': str(self.valid_df.shape) if self.valid_df is not None else 'None',
            'test': str(self.test_df.shape) if self.test_df is not None else 'None'
        }})
        
        ######################### global imputation based on train_df #########################
        # collect history for imputation_by_trainset
        hist2 = "Not Run"
        if len(list(self.train_df['__uid'].unique())) <= 50:
            print("Using 'mask' for predictor imputation (constant value -333) because too few subjects are sampled.")
            impute_input = "constant"
            print("Using 'mode' for response imputation because too few subjects are sampled.")
            impute_output = "most_frequent"
        # train_df not None to proceed, fillna based on training set only
        if self.train_df is not None:
            self.train_df_imputed = cohort_fillna(refer_df=self.train_df, df=self.train_df, vars=self.input_vars, method=impute_input, fill_value=fill_value, viz=viz)
            self.train_df_imputed = cohort_fillna(refer_df=self.train_df, df=self.train_df_imputed, vars=self.output_vars, method=impute_output, fill_value=fill_value, viz=viz)
            # both train_df and valid_df not None to proceed
            if self.valid_df is not None:
                self.valid_df_imputed  = cohort_fillna(refer_df=self.train_df, df=self.valid_df, vars=self.input_vars, method=impute_input, fill_value=fill_value, viz=viz)
                self.valid_df_imputed  = cohort_fillna(refer_df=self.train_df, df=self.valid_df_imputed , vars=self.output_vars, method=impute_output, fill_value=fill_value, viz=viz)
            # both train_df and test_df not None to proceed
            if self.test_df is not None:
                self.test_df_imputed = cohort_fillna(refer_df=self.train_df, df=self.test_df, vars=self.input_vars, method=impute_input, fill_value=fill_value, viz=viz)
                self.test_df_imputed = cohort_fillna(refer_df=self.train_df, df=self.test_df_imputed, vars=self.output_vars, method=impute_output, fill_value=fill_value, viz=viz)      
        print("Success! Project has updated attributes --- train_df_imputed, valid_df_imputed and test_df_imputed. ")
        hist2 = {'method': {'ml_inputs': impute_input, "ml_outputs": impute_output}}
        
        ######################### convert dataframe to TFDS #########################
        hist3 = "Not Run"
        if self.train_df_imputed is not None:
            if self.train_df_imputed.shape[0]>0:
                self.train_tfds = make_mvts_tfds_from_df(self.train_df_imputed, input_vars=self.input_vars, output_vars=self.output_vars, input_time_len=self.episode.input_time_len, output_time_len=self.episode.output_time_len, time_resolution=self.episode.time_resolution, time_lag=self.episode.time_lag, batch_size=batch_size)
        if self.valid_df_imputed is not None:
            if self.valid_df_imputed.shape[0]>0:
                self.valid_tfds = make_mvts_tfds_from_df(self.valid_df_imputed, input_vars=self.input_vars, output_vars=self.output_vars, input_time_len=self.episode.input_time_len, output_time_len=self.episode.output_time_len, time_resolution=self.episode.time_resolution, time_lag=self.episode.time_lag, batch_size=batch_size)
        if self.test_df_imputed is not None:
            if self.test_df_imputed.shape[0]>0:
                self.test_tfds = make_mvts_tfds_from_df(self.test_df_imputed, input_vars=self.input_vars, output_vars=self.output_vars, input_time_len=self.episode.input_time_len, output_time_len=self.episode.output_time_len, time_resolution=self.episode.time_resolution, time_lag=self.episode.time_lag, batch_size=batch_size)
        print("Success! Project has updated attributes --- train_tfds, valid_tfds and test_tfds. ")
        hist3 = {
            'batch_size': batch_size,
            'batch_shape': {
                'x': str([example_inputs.shape for example_inputs, _ in self.train_tfds.take(1)]),
                'y': str([example_labels.shape for _, example_labels in self.train_tfds.take(1)])
            }}
        # collect history
        self.hist_split_mvts = {'datetime':str(datetime.now())}
        self.hist_split_mvts.update({'split_dataframe':hist1})
        self.hist_split_mvts.update({'imputation':hist2})
        self.hist_split_mvts.update({'tensorflow_dateset':hist3})
        
        
        
    def extract_xy(self, shape_type="3d"):
        X_train=None
        Y_train=None
        X_valid=None
        Y_valid=None
        X_test=None
        Y_test=None
        X_train, Y_train = extract_xy(self.train_tfds, shape_type)
        X_valid, Y_valid = extract_xy(self.valid_tfds, shape_type)
        X_test, Y_test = extract_xy(self.test_tfds, shape_type)
        return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

    
    