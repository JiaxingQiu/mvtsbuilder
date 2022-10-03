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
from mvtsbuilder.functions.create_csv_pool import *
from mvtsbuilder.functions.cmp_src_csv import *
from mvtsbuilder.classes.episode import Episode
import random
from datetime import datetime
import os
import json
from importlib import resources

class Project:
    
    def __init__(self, work_dir):
        """Initiate a mvtsbuilder project, usually about a machine learning problem.

        Current working directory will be changed to given work_dir. 
        A meta_data folder will be automatedly created under work_dir if not exist. 
        Dictionary files including "variable_dict.json" and "csv_source_dict.json" will attempted to be read 
        from meta_data folder.

        Parameters
        ----------
        work_dir : str
            Path to your project working directory/foler.

        Returns
        -------
        mvtsbuilder.Project
            object with attributes and functions to engineer MVTS machine learning datasets.

        Examples
        --------
        >>> mvtsbuilder.Project("parent_folders/myproject")
        """
        

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
        self.df_raw_uid_ls = None
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
        assert os.path.exists(self.work_dir), 'Working directory not found: '+str(self.work_dir)
        os.chdir(self.work_dir)
        print('Working directory: ' + str(self.work_dir))
        # set meta data of dictionarys directory 
        self.meta_dir = str(self.work_dir) + '/meta_data'
        if not os.path.exists(self.meta_dir):
            os.mkdir(self.meta_dir)
        print('Meta_data directory: ' + str(self.meta_dir))
        # set variable_dict
        self.load_variable_dict()
        # set csv_source_dict
        self.load_csv_source_dict()

       
    def __str__(self):
        """Print a mvtsbuilder project and its working history.

        It's a good habit to print the mvtsbuilder project object after each step in the process, to monitor its updated attributes and function logs.
        The function calling history at a certain processing status will be documented and printed in dictionary format. 

        Parameters
        ----------
        
        Returns
        -------
        mvtsbuilder.Project.hist
            dict-like object with project object attributes and functions running log/history.

        Examples
        --------
        >>> myprj = mvtsbuilder.Project("parent_folders/myproject")
        >>> print(myprj)
        """
        
        self.hist = {
            'Project Info':{
                'datetime': str(self.datetime),
                'working_dir': str(self.work_dir),
                'meta_data_dir': str(self.meta_dir),
                'dictionary': {
                    'variable_dict': str(self.meta_dir)+'/variable_dict.json' if self.variable_dict is not None else str(self.meta_dir)+'/variable_dict.json not found',
                    'csv_source_dict': str(self.meta_dir)+'/csv_source_dict.json' if self.csv_source_dict is not None else str(self.meta_dir)+'/csv_source_dict.json not found'
                },
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

    def load_variable_dict(self):
        """Read variable dictionary object as an attribute of mvtsbuilder project.
        
        Attribute variable_dict will be open and read from variable_dict.json file under work_dir/meta_data folder.
        Attribute variable_dict has information of each variable to be used in a machine learning problem, 
        such as its type(numeric/factor), how it should be imputed, its outlier cutoffs, the final unified name if 
        multiple sources of data will be used. 
        
        Parameters
        ----------
        
        Returns
        -------
        mvtsbuilder.Project.variable_dict
            dict-like object of variable information involved in current project, usually a machine learning or data processing problem.
        mvtsbuilder.Project.input_vars
            list of explanatory variable names as input of a model
        mvtsbuilder.Project.output_vars
            list of responce variable names as output of a model
        mvtsbuilder.Project.input_vars_byside
            list of candidate explanatory variable names to be engineered but not included as the input of a ML model.
        mvtsbuilder.Project.output_vars_byside
            list of candidate responce variable names to be engineered but not included as the output of n a ML model.
        
        Examples
        --------
        >>> myprj = mvtsbuilder.Project("parent_folders/myproject")
        >>> myprj.load_variable_dict()
        >>> print(myprj)
        """
        try:
            fullname = str(self.meta_dir)+'/variable_dict.json'
            f = open(fullname, "r")
            self.variable_dict = json.loads(f.read())
            print("Project variable dictionary loaded;")
            print(json.dumps(self.variable_dict, indent=2))
        except:
            print("--- Project variable_dict.json not exist. ---")
            print("You can put a previous 'variable_dict.json' file in path '"+str(self.work_dir)+"/meta_data'.")
            print("Or, You can use function .new_demo_variable_dict() to create one. Please modify the newly created file '"+str(self.work_dir)+"/meta_data/demo_variable_dict_TIMESTAMP.json' and save it as '"+str(self.work_dir)+"/meta_data/variable_dict.json';")
        if self.variable_dict is not None:
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
    

    def load_csv_source_dict(self):
        """Read csv sources dictionary object as an attribute of mvtsbuilder project.
        
        Attribute csv_source_dict will be open and read from csv_source_dict.json file under work_dir/meta_data folder.
        Attribute csv_source_dict has information of each csv format data from different sources to be used in a machine learning problem, 
        such as its file path, source institute name, what major content it's about, how it should be merged with other sources.
        
        Parameters
        ----------
        
        Returns
        -------
        mvtsbuilder.Project.csv_source_dict
            dict-like object of csv data source information involved in current project, usually a machine learning or data processing problem.
        
        Examples
        --------
        >>> myprj = mvtsbuilder.Project("parent_folders/myproject")
        >>> myprj.load_csv_source_dict()
        >>> print(myprj)
        """

        try:
            fullname = str(self.meta_dir)+'/csv_source_dict.json'
            f = open(fullname, "r")
            self.csv_source_dict = json.loads(f.read())
            print("Project csv_source dictionary loaded;")
            print(json.dumps(self.csv_source_dict, indent=2))
        except:
            print("--- Project csv_source_dict.json not exist. ---")
            print("You can put a previous 'csv_source_dict.json' file in path '"+str(self.work_dir)+"/meta_data'.")
            print("Or, You can use function .new_demo_csv_source_dict() to create one. Please modify the newly created file '"+str(self.work_dir)+"/meta_data/demo_csv_source_dict_TIMESTAMP.json' and save it as '"+str(self.work_dir)+"/meta_data/csv_source_dict.json';")
    
    def def_episode(self, input_time_len, output_time_len, time_resolution=None, time_lag=None, anchor_gap=None):
        """Define Episode for a MVTS project.
        
        Episode is a key temporal concept in a Multi-Varaible Time Series (MVTS) project, that's prerequisite in other mvtsbuilder functionailties. 
        Project.episode need to be defined to let mvtsbuilder know the temporal structure of your dataset.

        Parameters
        ----------
        input_time_len: float
            time length before an anchor of episode to use as input in ML, in raw temporal scale defined in variable_dict
        output_time_len: float
            time length after an anchor of episode to use as output in ML, in raw temporal scale defined in variable_dict
        time_resolution: float
            how long in raw scale is aggregated into a unit in final episode, i.e. if your raw data is in minutes, you want to convert it to an hour per row, 60 should be given here.
        time_lag: float
            lag of time before input(X) chunk and output(y) chunk within an episode if any, in otehr words, how long you want to forecast ahead of the outcome. 
        anchor_gap: float
            minimum length of time (gap) between 2 episodes in raw scale. 

        Returns
        -------
        mvtsbuilder.Project.episode
            dict-like object of  the definition of episide for current machine learning setting

        Examples
        --------
        >>> myprj = mvtsbuilder.Project("parent_folders/myproject")
        >>> myprj.def_episode(input_time_len=4*24*60,output_time_len=1*24*60, time_resolution=60, time_lag=0, anchor_gap=7*24*60)
        >>> print(myprj)
        """
        
        # relaod dictionaries
        self.load_variable_dict()
        # make sure Project has variable_dict object at hand by now
        if self.variable_dict is None: 
            return
        # initiate episode instance
        self.episode = Episode(input_time_len, output_time_len, self.variable_dict['__time']['unit'], time_resolution=time_resolution, time_lag=time_lag, anchor_gap=anchor_gap)
        # collect success history dictionary
        print("Success! Project has updated attributes --- episode. ")
        self.hist_def_episode = {'datetime': str(datetime.now())}
        self.hist_def_episode.update(self.episode.__dict__)

         
    def build_mvts(self, source=None, nsbj=None, frac=0.3, replace=True, stratify_by=None, skip_uid=None, keep_uid=None, return_episode=True, topn_eps=None, dummy_na=False, sep="---", viz=False, viz_ts=False):
        """Build episode-wise Multi-Variable Time Series DataFrame.
        
        
        Parameters
        ----------
        source: string or pandas.core.frame.DataFrame
            the source of data, supporting formats include a pandas dataframe, a string of the path to the csv_pool directory.
        nsbj: int
            number of subjects to sample from the source, it can be None.
        frac: float
            sampling rate or fraction of subjects from the source, not work if nsbj is given.
        replace: bool
            sampling subjects from source data with or without replacement, True means with replacement.
        stratify_by: list
            list of final variable names by which, sampling should be stratefied by.
        skip_uid: list
            list of subject IDs that user specified to skip engineering.
        keep_uid: list
            list of subject IDs that user specified to engieer.
        return_episode: bool
            whether or not an episode-wise MVTS dataframe mvts_df should be returned, if False, only subject-wise dataframe sbj_df will be returned.
        topn_eps: int
            keep first N episode per subject.
        sep: str
            special separating string used in csv_pool naming fashion, default is "---". 
        dummy_na: bool
            [under construction] whether or not a column of indicator of NA should be created for each variable.
        viz: bool
            [under improvement] whether or not to visualize distribution of each variable, and their stratified distribution by outcome
        viz_ts: bool
            [under improvement] whether or not to visualize time series of each variable, and their stratified time series trend by outcome


        Returns
        -------
        mvtsbuilder.Project.mvts_df
            pandas.DataFrame object of multi-variable time series data of stacked episodes. 
        mvtsbuilder.Project.sbj_df
            pandas.DataFrame object of subject-wise data that's been sampled, cleaned and engineered. 
        mvtsbuilder.Project.sample_info
            string of the information of sample size, and cohort size in terms of enrolled subjects.

        Examples
        --------
        >>> import pandas as pd
        >>> myprj = mvtsbuilder.Project("parent_folders/myproject")
        >>> df = pd.read_csv(".../yourdata.csv")
        >>> myprj.build_mvts(source=df, nsbj=300)
        >>> myprj.build_mvts(source="csv_pool_path", nsbj=100, replace=True, sep='_')
        >>> print(myprj)
        """

        # make sure Project has episode object at hand by now
        if self.episode is None:
            return 'No episode defined -- you can use def_episode() to define one'
        episode = self.episode


        # interpret source as csv_pool_dir or df_raw
        csv_pool_dir = None # the path to csv_pool directory
        df_raw = None # raw dataframe object
        if str(type(source)).__eq__("<class 'str'>"):
            csv_pool_dir = source
        elif str(type(source)).__eq__("<class 'pandas.core.frame.DataFrame'>"):
            df_raw = source
        else:
            return "Parameter 'source' must be one of these types: <class 'str'> / <class 'pandas.core.frame.DataFrame'>"
        
        # fix argument nsbj and frac
        if nsbj is not None:
            if str(type(nsbj)).__eq__("<class 'int'>"):
                if int(nsbj) >= 1:
                    nsbj = int(nsbj)
                    frac = 0.3 # set as default
                else:
                    return "Parameter 'nsbj' must be >= 1!"
            else:
                return "Parameter 'nsbj' must be int type or None!"
        else:
            if str(type(frac)).__eq__("<class 'float'>"):
                if float(frac)>0 and float(frac)<=1:
                    frac = float(frac)
                else:
                    return "Parameter 'frac' must range in (0, 1]"
            else:
                return "Parameter 'frac' must be float type!"
            
        
        hist1 = "Not Run" # sampling
        ################# generate MVTS from df_raw #################
        # filtering raw dataframe by requests
        if df_raw is not None:
            print("Project is Projecting customized table format data ---")
            hist1 = {'from': 'pandas dataframe object'}
            # find __uid column in the raw data
            uid_col = list(set(self.variable_dict['__uid']['src_names']).intersection(set(df_raw.columns)))[0]
            uid_ls_raw = list(df_raw[uid_col].unique()) # all the uid list from original df_raw
            ## initiate all_uid list from scratch or sampling history
            if self.df_raw_uid_ls is None:
                all_uid = uid_ls_raw
            else:
                assert len(self.df_raw_uid_ls)>0, "No subject left to be been sampled from!"
                all_uid = self.df_raw_uid_ls
            ## trim all_uid list
            # skip uid in "skip_uid list"
            all_uid2 = all_uid
            if skip_uid is not None:
                try:
                    all_uid2 = list(set(all_uid) - set(skip_uid))
                except:
                    print("Failed to skip given uid list, please check input")
            # only keep uid in "keep_uid list"
            if keep_uid is not None:
                try:
                    all_uid2 = list(set(all_uid) & set(keep_uid))
                except:
                    print("Failed to only keep given uid list, please check input")
            all_uid = all_uid2
            # find ksbj, which is the final number of subjects to be sampled from all_uid
            if nsbj is None:
                # if user only specify fraction, it should always be relative to the overall population
                ksbj = int(len(uid_ls_raw)*frac)
            else:
                # ksbj cannot be greater than the length of all_uid
                ksbj = min(int(nsbj), int(len(all_uid)))
            ksbj = min(int(ksbj), int(len(all_uid)))
            # recaculate frac_sbj
            frac_sbj = ksbj/int(len(all_uid))
            # stratified sampling
            all_uid3 = np.array(all_uid) # convert list to np.array
            if stratify_by is None:
                all_uid3 = list(all_uid3[list(random.sample(list(range(len(all_uid3))), ksbj))])
            else:
                stratify_by = list(stratify_by)
                stratify_by = list(set(stratify_by).intersection(set(self.variable_dict.keys())))
                stratify_by_list = [var for var in stratify_by if 'factor' in self.variable_dict[var].keys()]
                print("--- Stratify sampling by :" + str(stratify_by_list))
                # find all raw columns in df_raw columns that intersects stratify_by (i.e. ['y', 'txp'])
                colnames = []
                for var in stratify_by_list: 
                    colnames = colnames + list(self.variable_dict[var]['src_names'])
                colnames = list(set(colnames).intersection(set(df_raw.columns)))
                all_uid3 = list(df_raw.groupby(colnames)[uid_col].apply(lambda x: x.sample(frac=frac_sbj)).reset_index(drop=True).unique())
            # final filtered df_raw
            df_raw = df_raw.loc[df_raw[uid_col].isin(list(all_uid3)),:]
            # update global df_raw_uid_ls that keeps track of sampling info from raw dataframe
            # allow sampling with/without replacement
            if replace:
                self.df_raw_uid_ls = uid_ls_raw
            else:
                self.df_raw_uid_ls = list(set(uid_ls_raw) - set(all_uid3))

        ################# generate MVTS from csv_pool ################# 
        if csv_pool_dir is not None:
            print("Project is sampling from csv_pool --- ")
            hist1 = {'from': 'csv_pool', 'path': str(csv_pool_dir), 'sep': str(sep)}
            print("csv_pool_dir: "+str(csv_pool_dir))
            # reload csv_source_dict
            self.load_csv_source_dict()
            # stop if csv_source dict is not ready
            if self.csv_source_dict is None:
                return
            # sampling from csv_pool
            if self.df_csv_fullname_ls is None:
                self.df_csv_fullname_ls = init_csv_fullname_ls(csv_pool_dir, sep=sep)
            if replace:
                self.df_csv_fullname_ls = init_csv_fullname_ls(csv_pool_dir, sep=sep)
            
            
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

            print("Success! Project has updated attributes --- mvts_df, sbj_df, sample_info")
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

    
    def new_demo_variable_dict(self):
        # load demo.json file from package data
        with resources.path("mvtsbuilder.data", "demo_variable_dict.json") as f:
            from_path = f
        json_obj = open(from_path, "r")
        # read dict_demo in dictionary object
        dict_demo = json.loads(json_obj.read())
        # save it to meta_data folder under user project working directory
        with open(self.meta_dir + '/demo_variable_dict_'+str(datetime.now())+'.json', 'w') as f:
            json.dump(dict_demo, f)
        print("A new demo of variable dictionary is ready for you, using path: \n")
        return str(self.meta_dir) + '/demo_variable_dict_'+str(datetime.now())+'.json'
    
    
    def new_demo_csv_source_dict(self):
        # load demo.json file from package data
        with resources.path("mvtsbuilder.data", "demo_csv_source_dict.json") as f:
            from_path = f
        json_obj = open(from_path, "r")
        # read dict_demo in dictionary object
        dict_demo = json.loads(json_obj.read())
        # save it to meta_data folder under user project working directory
        with open(self.meta_dir + '/demo_csv_source_dict_'+str(datetime.now())+'.json', 'w') as f:
            json.dump(dict_demo, f)
        print("A new demo of csv source dictionary is ready for you, using path: \n")
        return str(self.meta_dir) + '/demo_csv_source_dict_'+str(datetime.now())+'.json'
      
    def create_csv_pool(self, csv_pool_dir=None, overwrite=False, source_key=None, file_key=None, sep="---"):
        import os
        if csv_pool_dir is None: # set default csv chunk pool dir
            csv_pool_dir = self.work_dir + '/csv_pool'
            if not os.path.exists(csv_pool_dir):
                os.mkdir(csv_pool_dir)
            else:
                if overwrite:
                    print('you are overwriting csv_pool in dir -- ' + csv_pool_dir)
                else: 
                    print(str(csv_pool_dir) + ' already exist, you can remove the folder or set overwrite=True')
                    return
        self.load_csv_source_dict()
        self.load_variable_dict()
        assert self.csv_source_dict is not None, 'csv_source_dict.json not exist!'
        assert self.variable_dict is not None, 'variable_dict.json not exist!'
        create_csv_pool(self.csv_source_dict, self.variable_dict, csv_pool_dir, source_key=source_key, file_key=file_key, sep=sep)
      
    def cmp_src_csv(self, nrows=None, var_list=None):
        self.load_csv_source_dict()
        self.load_variable_dict()
        assert self.csv_source_dict is not None, 'csv_source_dict.json not exist!'
        assert self.variable_dict is not None, 'variable_dict.json not exist!'
        cmp_src_csv(self.csv_source_dict, self.variable_dict, nrows=nrows, var_list=var_list)
 