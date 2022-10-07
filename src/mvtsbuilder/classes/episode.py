""" Episode class. 

    Module description details:
    Episode object contains all the required definitions for a medical episode / episode of some outcome,
    before creating multivarable time series datasets, (flat dataframes or 3D TFDS), user must input 
    an episode definition by initiating an episode object.
    
    Usage example:
    >>> Episode(input_time_len=4*24*60,output_time_len=1*24*60, time_resolution=60, time_lag=0, anchor_gap=7*24*60)

"""
import json

class Episode:

    def __init__(self, input_time_len, output_time_len, time_unit=None, time_resolution=None, time_lag=None, anchor_gap=None):
        """ 
        initiate an episode object to set the definition of an episode for the time series study
    
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
        object instance of Episode class
            dict-like object of the definition of episide for current machine learning setting

        """
        
        self.time_resolution = None
        self.input_time_len = None
        self.output_time_len = None
        self.time_lag = None
        self.anchor_gap = None

        if time_unit is None:
            time_unit = "(time unit not specified)"
        self.time_unit = str(time_unit)
        if time_resolution is None:
            time_resolution = 1
        if time_lag is None:
            time_lag = 0
        
        assert int(time_resolution)>0, "time resolution must be integar greater than 0"
        self.time_resolution = int(time_resolution)
        assert int(input_time_len)>0, "input time length must be integar greater than 0"
        self.input_time_len = int(input_time_len)
        assert int(output_time_len)>0, "output time length must be integar greater than 0"
        self.output_time_len = int(output_time_len)
        assert int(time_lag)>=0, "time lag/gap between last input and first output should be greater than or equal to 0"
        self.time_lag = int(time_lag)
        
        if anchor_gap is None:
            self.anchor_gap = self.input_time_len + self.time_lag + self.output_time_len
        else:
            self.anchor_gap = anchor_gap
        
        
    
    def __str__(self):
        return '\n'.join([
            f' ',
            f'An episode is defined to ',
            f'--- use {self.input_time_len} {self.time_unit}(s) long input variables ',
            f'--- predict {self.output_time_len} {self.time_unit}(s) response variables into the future',
            f'--- lag {self.time_lag} {self.time_unit}(s) between predictors and responses',
            f'--- increase by every {self.time_resolution} {self.time_unit}(s)',
            f'--- last at most {self.anchor_gap} {self.time_unit}(s) long'
        ])
    
    def document(self):
        # document episode definition as json file under meta_data folder
        with open("./meta_data/episode.json", "w") as outfile:
            json.dump(self.__dict__, outfile)
            print("episode.json file is saved under meta_data folder")
    
    def show(self):
        json_object = json.dumps(self.__dict__, indent = 4) 
        print(json_object)


        