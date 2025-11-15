# Plotter

A Python library for plotting time-series and categorical data with a structured, extensible approach.
Just helps to make my life easier for most basic plots.

## Installation

pip install -e .


## Usage

```
from plotter import DataPlotter

plotter = DataPlotter(df, time_col='datetime', category_col1='agent', category_col2='topic') plotter.countplot('agent') plotter.temporal.count_by_period(period='h')
```
For more usage see `plotter_usage.ipynb`

