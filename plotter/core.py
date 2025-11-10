
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from arviz import hdi
from .temporal import TemporalPlots

class DataPlotter:
    """Main plotting class for time-series and categorical data visualization.

    Args:
        df: DataFrame containing the data to plot
        time_col: Name of the datetime column
        category_col1: Primary categorical column (e.g., 'agent')
        category_col2: Secondary categorical column (e.g., 'topic')
        time_features: List of time features to extract from datetime column
        figsize: Default figure size as (width, height) tuple
    """
    def __init__(self, df, time_col='datetime', category_col1=None, category_col2=None, 
        time_features=['hour', 'day', 'dayofweek', 'week', 'year'], figsize=(7, 7)):
        self.df = df
        self.time_col = time_col
        self.category_col1=category_col1
        self.category_col2=category_col2
        self.time_features=time_features
        self.figsize=figsize
        if self.time_col is not None:
            self._get_time_features()
        self.callbacks = {
            "pre":[],
            "post":[]
            #'pre': [lambda ax: ax.set_yscale('log')],  # before plotting
            #'post': [lambda ax: ax.axhline(y=10, color='r')] # after plotting
        }
        self.temporal = TemporalPlots(self)

    def _get_time_features(self):
        """Extract time features (hour, day, dayofweek, etc.) from datetime column."""
        self.df[self.time_col]=pd.to_datetime(self.df[self.time_col])
        for time_feature in self.time_features:
            if time_feature=='week':
                self.df[time_feature] = self.df[self.time_col].dt.isocalendar().week
            elif time_feature == 'dayofweek':
                self.df[time_feature] = self.df[self.time_col].dt.day_name()
            else:
                self.df[time_feature] = getattr(self.df[self.time_col].dt, time_feature)

    def _set_title_labels(self, ax, title=None, xlabel=None, ylabel=None):
        """Set title and axis labels on the given axes."""
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        return ax

    def _set_figsize(self, figsize=None):
        """Apply callbacks (pre or post) to modify axes."""
        figsize= self.figsize if figsize is None else figsize
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    def _apply_callbacks(self, ax, callbacks, stage):
        """Apply callbacks (pre or post) to modify axes."""
        for callback in self.callbacks.get(stage, []):
            callback(ax)
        if callbacks is not None:
            for callback in callbacks.get(stage, []):
                callback(ax)
        return ax

    def _create_plot(self, plot_func, figsize=None, title=None, xlabel=None, ylabel=None, callbacks=None):
        """Create a single plot with callbacks and labels."""
        fig, ax = self._set_figsize(figsize)
        ax=self._apply_callbacks(ax, callbacks, 'pre')
        ax = plot_func(ax)
        ax=self._apply_callbacks(ax, callbacks, 'post')
        ax = self._set_title_labels(ax, title, xlabel, ylabel)
        return fig, ax

    def _create_facet_plot(self, nrows, ncols, plot_func, data_subsets,
                       figsize=None, subplot_titles=None, 
                       suptitle=None, xlabel=None, ylabel=None, 
                       callbacks=None):
        """Create a faceted plot with multiple subplots.

        Args:
            nrows: Number of rows in subplot grid
            ncols: Number of columns in subplot grid
            plot_func: Function that takes (ax, data) and plots on ax
            data_subsets: List of data subsets, one per subplot
            subplot_titles: List of titles for each subplot
            suptitle: Overall figure title
        """
        figsize = self.figsize if figsize is None else figsize
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()

        # Loop through subplots
        for idx, (ax, data) in enumerate(zip(axes, data_subsets)):
            # Apply pre-callbacks
            ax = self._apply_callbacks(ax, callbacks, 'pre')

            # Call plot_func with this ax and data
            ax = plot_func(ax, data)

            # Apply post-callbacks
            ax = self._apply_callbacks(ax, callbacks, 'post')

            # Set subplot title if provided
            if subplot_titles and idx < len(subplot_titles):
                ax.set_title(subplot_titles[idx])

        # Hide unused subplots
        n_subsets = len(data_subsets)
        for idx in range(n_subsets, len(axes)):
            axes[idx].set_visible(False)

        # Set overall title if provided
        if suptitle:
            fig.suptitle(suptitle)

        # Set common labels if provided
        if xlabel:
            fig.supxlabel(xlabel)
        if ylabel:
            fig.supylabel(ylabel)

        plt.tight_layout()
        return fig, axes

    def countplot(self, group_col=None, title=None, xlabel=None, ylabel=None, kind='barh', figsize=None, top_n=None, callbacks=None):
        """Plot value counts for a categorical column.

        Args:
            group_col: Column to count values for
            kind: Plot type ('barh', 'bar', 'pie', etc.)
            top_n: Show only top N values
        """
        if group_col is None:
                group_col=self.category_col1
        def plot_func(ax):
            ax = self.df[group_col].value_counts().sort_values()[:top_n].plot(kind=kind, ax=ax)
            return ax

        title = f"Count of {group_col}" if title is None else title
        return self._create_plot(plot_func, figsize, title, xlabel, ylabel, callbacks) 

    def crosstab_heatmap(self, category_col1=None, category_col2=None, title=None, xlabel=None, ylabel=None, 
        figsize=None, cmap='Blues', callbacks=None, **kwargs):
        """Plot heatmap of crosstab between two categorical columns.

        Args:
            category_col1: First categorical column
            category_col2: Second categorical column
            cmap: Colormap for heatmap
            **kwargs: Additional arguments passed to sns.heatmap
        """
        category_col1= self.category_col1 if category_col1 is None else scategory_col1
        category_col2= self.category_col2 if category_col2 is None else scategory_col2

        def plot_func(ax):
            matrix=pd.crosstab(self.df[category_col1], self.df[category_col2])
            ax=sns.heatmap(matrix,  cmap=cmap, **kwargs)
            return ax

        return self._create_plot(plot_func, figsize, title, xlabel, ylabel, callbacks) 

    def category_mean_with_hdi(self, value_col, category_col=None, 
                          stat='mean', show_hdi=True, hdi_prob=0.94,
                          kind='barh', title=None, xlabel=None, ylabel=None,
                          linestyle='', figsize=None, top_n=None, callbacks=None):
        """Plot mean/median of value_col per category with HDI error bars.

        Args:
            value_col: Column containing values to aggregate
            category_col: Categorical column to group by
            stat: 'mean' or 'median'
            show_hdi: Whether to show HDI error bars
            hdi_prob: HDI probability (default 0.94)
            top_n: Show only top N categories by count
        """
        if category_col is None:
            category_col=self.category_col1

        def plot_func(ax):
            grouped = self.df.groupby(category_col)[value_col]
            if stat=='mean':
                means = grouped.mean() 
            elif stat=='median':
                means=grouped.median()

            lower_bounds = []
            upper_bounds = []
            for category in means.index:
                values = self.df[self.df[category_col] == category][value_col].values
                interval = hdi(values, hdi_prob=hdi_prob)
                lower_bounds.append(interval[0])
                upper_bounds.append(interval[1])

            ax.errorbar(x=means.values, y=means.index, 
                xerr=[lower_bounds, upper_bounds],
                fmt='o', linestyle=linestyle)
            return ax

        title = f"{stat} of per {category_col}" if title is None else title
        return self._create_plot(plot_func, figsize, title, xlabel, ylabel, callbacks)
