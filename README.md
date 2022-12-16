# mvtsbuilder

Build Multi-variable Time Series (MVTS) pandas dataframe or tensorflow dataset, by uniforming multiple sources of raw data in csv format or database format to use in the same ML project.

## Installation

```bash
$ pip install mvtsbuilder
```

## Usage

'mvtsbuilder' can build Multi-variable Time Seris data in pandas DataFrame or Tensorflow Dataset format, to use in Machine Learning or Deep Learning tools. 
Usage is as follows:

**Create your mvts project**<br> 

```python
import mvtsbuilder

work_dir = "your project folder directory" # path to your project folder
prj = mvtsbuilder.Project(work_dir)
print(prj)
```

**Put dictionary in place**<br> 

- 'variable_dict.json' need to be prepared in "meta_data" folder under your working directory. 
- 'csv_source_dict.json' need to be prepared in "meta_data" folder under your working directory if you are sampling data from csv_pool.

```python
prj.new_demo_variable_dict()
prj.new_demo_csv_source_dict()
print(prj)
```

**Define Episode**<br> 

```python
prj.def_episode(
    input_time_len=4*24*60,
    output_time_len=1*24*60, 
    time_resolution=60, 
    time_lag=0, 
    anchor_gap=7*24*60)
print(prj)

```
**Build MVTS DataFrame**<br> 

```python
# from raw DataFrame object
import pandas as pd

df = pd.read_csv("PATH_TO_YOUR_CSV.csv")
prj.build_mvts(source=df)
print(prj)

# from csv_pool
csv_pool_path = 'PATH_TO_YOUR_CSV_POOL'
prj.build_mvts(
    source = csv_pool_path, 
    nsbj = 1000, 
    sep = '_')
print(prj)
```
**Split MVTS DataFrame to ML DF and TFDS**<br> 

```python
prj.split_mvts(
    valid_frac = 0.2, 
    test_frac = 0.1, 
    impute_input='median', # imputation on predictors
    impute_output='none',# imputation on response (no need in BSI project)
    byepisode = True, 
    batch_size = 64)
print(prj)
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`mvtsbuilder` was created by Jiaxing Qiu. It is licensed under the terms of the MIT license.

## Credits

`mvtsbuilder` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
