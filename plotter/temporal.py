
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from arviz import hdi

class TemporalPlots:
    """Time-series plotting methods."""
    def __init__(self, parent):
        self.parent = parent
    
    def count_by_period(self, period='D', kind='line', title=None, xlabel=None, ylabel=None, figsize=None, callbacks=None):
        """Plot event counts aggregated by time period.
        
        Args:
            period: Pandas frequency string ('D', 'h', 'W', etc.)
            kind: Plot type ('line', 'bar', etc.)
        """
        def plot_func(ax):
            counts = self.parent.df.groupby(pd.Grouper(key=self.parent.time_col, freq=period)).size()
            counts.plot(kind=kind, ax=ax)
            return ax
        
        title = f'Count per period {period}' if title is None else title
        xlabel =f'period {period}' if xlabel is None else xlabel
        ylabel =f'Count' if ylabel is None else ylabel
    
        return self.parent._create_plot(plot_func, figsize, title, xlabel, ylabel, callbacks) 
        

    def count_by_period_and_category(self, period='D', category_col=None, kind='line', 
                                  title=None, xlabel=None, ylabel=None, 
                                  figsize=None, callbacks=None):
        """Plot event counts by time period, split by category.
        
        Args:
            period: Pandas frequency string ('D', 'h', 'W', etc.)
            category_col: Column to split counts by
            kind: Plot type ('line', 'bar', etc.)
        """
        category_col = self.parent.category_col1 if category_col is None else category_col

        def plot_func(ax):
            counts = self.parent.df.groupby([pd.Grouper(key=self.parent.time_col, freq=period), category_col]).size()
            ax=counts.unstack().plot(kind=kind, ax=ax)
            return ax
        
        title=f"Count of {category_col} per period of {period}"
        xlabel=f"Period {period}" if xlabel is None else xlabel
        ylabel="Count" if ylabel is None else ylabel
        return self.parent._create_plot(plot_func, figsize, title, xlabel, ylabel, callbacks)

    def _plot_mean_by_period(self, ax, data, value_col, period, line_col, 
                            stat='mean', show_hdi=False, hdi_prob=0.94, 
                            alpha=0.3, kind='line'):
        """Internal method to plot mean/median by period with optional HDI bands."""
        grouped = data.groupby([
            pd.Grouper(key=self.parent.time_col, freq=period), 
            line_col])[value_col]
    
        if stat == 'mean':
            values = grouped.mean().unstack()
        elif stat == 'median':
            values = grouped.median().unstack()
        
        values.plot(kind=kind, ax=ax)
        lines = ax.get_lines()
        color_map = {line.get_label(): line.get_color() for line in lines}
        
        if show_hdi:
            for category in values.columns:
                lower_bounds, upper_bounds, time_points = [], [], []
                
                for time_point in values.index:
                    subset = data[
                        (data[self.parent.time_col] >= time_point) & 
                        (data[self.parent.time_col] < time_point + pd.Timedelta(1, unit=period)) &
                        (data[line_col] == category)
                    ][value_col].values
                    
                    if len(subset) > 1:
                        hdi_interval = hdi(subset, hdi_prob=hdi_prob)
                        lower_bounds.append(hdi_interval[0])
                        upper_bounds.append(hdi_interval[1])
                        time_points.append(time_point)
                    elif len(subset) == 1:
                        warnings.warn(f"Only one observation for {category} at {time_point}, skipping HDI")
                
                if time_points:
                    ax.fill_between(time_points, lower_bounds, upper_bounds, 
                        alpha=alpha, color=color_map[category])
        return ax

    def mean_by_period_and_category(self, value_col, period='D', category_col=None, 
                            stat='mean', show_hdi=False, hdi_prob=0.94, 
                            kind='line', title=None, xlabel=None, ylabel=None, 
                            figsize=None, callbacks=None, alpha=0.3):
        """Plot mean/median of a value column by time period and category.
        
        Args:
            value_col: Column containing values to aggregate
            period: Pandas frequency string ('D', 'h', 'W', etc.)
            category_col: Column to split lines by
            stat: 'mean' or 'median'
            show_hdi: Whether to show HDI uncertainty bands
            hdi_prob: HDI probability (default 0.94)
            alpha: Transparency of HDI bands
        """
        category_col = self.parent.category_col1 if category_col is None else category_col
        
        def plot_func(ax):
            ax = self._plot_mean_by_period(ax, self.parent.df, value_col, period, category_col, 
                            stat, show_hdi, hdi_prob, alpha, kind)
            return ax
        
        title = f"{stat.capitalize()} of {value_col} by {category_col} per period {period}" if title is None else title
        xlabel = f"Period {period}" if xlabel is None else xlabel
        ylabel = f"{stat.capitalize()} {value_col}" if ylabel is None else ylabel
        
        return self.parent._create_plot(plot_func, figsize, title, xlabel, ylabel, callbacks)


    def mean_by_period_and_categories(self, value_col, period='D', 
                                    line_col=None, facet_col=None,
                                    stat='mean', show_hdi=False, hdi_prob=0.94, 
                                    kind='line', ncols=3, title=None, xlabel=None, ylabel=None, 
                                    figsize=None, callbacks=None, alpha=0.3, legend='right'):
        """Plot mean/median by time period in faceted subplots.
        
        Args:
            value_col: Column containing values to aggregate
            period: Pandas frequency string ('D', 'h', 'W', etc.)
            line_col: Column to split lines by within each subplot
            facet_col: Column to create separate subplots for
            stat: 'mean' or 'median'
            show_hdi: Whether to show HDI uncertainty bands
            ncols: Number of columns in facet grid
            legend: Legend position ('right', 'bottom', 'top', 'left', or None)
        """
        line_col = self.parent.category_col1 if line_col is None else line_col
        facet_col = self.parent.category_col2 if facet_col is None else facet_col
        
        facet_values = self.parent.df[facet_col].unique()
        data_subsets = [self.parent.df[self.parent.df[facet_col] == fv] for fv in facet_values]
        subplot_titles = list(facet_values)
        
        n_facets = len(facet_values)
        nrows = (n_facets + ncols - 1) // ncols
        
        def plot_func(ax, data):
            return self._plot_mean_by_period(ax, data, value_col, period, line_col,
                                            stat, show_hdi, hdi_prob, alpha, kind)
        
        fig, axes = self.parent._create_facet_plot(nrows, ncols, plot_func, data_subsets,
                                                figsize, subplot_titles, title, xlabel, ylabel, callbacks)
        
        # Handle legend
        if legend is not None:
            handles, labels = axes[0].get_legend_handles_labels()
            for ax in axes:
                legend_obj = ax.get_legend()
                if legend_obj:
                    legend_obj.remove()
            
            if legend == 'right':
                fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
            elif legend == 'bottom':
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(labels))
            elif legend == 'top':
                fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(labels))
            elif legend == 'left':
                fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0, 0.5))
        else:
            # Remove all legends
            for ax in axes:
                legend_obj = ax.get_legend()
                if legend_obj:
                    legend_obj.remove()

        return fig, axes
