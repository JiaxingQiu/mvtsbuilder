from mvtsbuilder.classes.engineer import Engineer
from mvtsbuilder.classes.querier import Querier



class Project:
    
    def __init__(self, work_dir):
        self.querier = Querier(work_dir)
        self.engineer = Engineer(work_dir)
