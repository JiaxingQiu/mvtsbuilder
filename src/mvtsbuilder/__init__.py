# read version from installed package
from importlib.metadata import version
__version__ = version("mvtsbuilder")

# populate package namespace
from mvtsbuilder.classes.project import Project