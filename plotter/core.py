
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
                            kind='dot', title=None, xlabel=None, ylabel=None,
                            alpha=0.3, figsize=None, callbacks=None):
        """Plot mean/median of value_col per category with HDI.

        Args:
            value_col: Column containing values to aggregate
            category_col: Categorical column (e.g., 'agent')
            stat: 'mean' or 'median'
            show_hdi: Whether to show HDI error bars/bands
            hdi_prob: HDI probability (default 0.94)
            kind: 'dot' (with error bars) or 'line' (with shaded area)
            alpha: Transparency for shaded HDI area
        """
        category_col = self.category_col1 if category_col is None else category_col

        def plot_func(ax):
            categories = self.df[category_col].unique()
            means = []
            lower_bounds = []
            upper_bounds = []

            for cat in categories:
                values = self.df[self.df[category_col] == cat][value_col].values
                if stat == 'mean':
                    means.append(np.mean(values))
                elif stat == 'median':
                    means.append(np.median(values))

                if show_hdi and len(values) > 1:
                    interval = hdi(values, hdi_prob=hdi_prob)
                    lower_bounds.append(interval[0])
                    upper_bounds.append(interval[1])
                else:
                    lower_bounds.append(means[-1])
                    upper_bounds.append(means[-1])

            x = range(len(categories))

            if kind == 'dot':
                ax.errorbar(x, means, 
                        yerr=[[m - l for m, l in zip(means, lower_bounds)],
                                [u - m for m, u in zip(means, upper_bounds)]],
                        fmt='o', capsize=5)
                ax.set_xticks(x)
                ax.set_xticklabels(categories)
            elif kind == 'line':
                ax.plot(x, means, marker='o')
                if show_hdi:
                    ax.fill_between(x, lower_bounds, upper_bounds, alpha=alpha)
                ax.set_xticks(x)
                ax.set_xticklabels(categories)

            return ax

        title = f"{stat} of {value_col} by {category_col}" if title is None else title
        xlabel = category_col if xlabel is None else xlabel
        ylabel = f"{stat} {value_col}" if ylabel is None else ylabel

        return self._create_plot(plot_func, figsize, title, xlabel, ylabel, callbacks)

    def category_mean_by_group(self, value_col, category_col=None, group_by=None,
                            stat='mean', show_hdi=True, hdi_prob=0.94,
                            kind='line', title=None, xlabel=None, ylabel=None,
                            alpha=0.3, figsize=None, callbacks=None):
        """Plot mean/median of value_col per category across groups with HDI.

        Args:
            value_col: Column containing values to aggregate
            category_col: Categorical column (e.g., 'agent')
            group_by: Grouping column (e.g., 'dayofweek')
            stat: 'mean' or 'median'
            show_hdi: Whether to show HDI error bars/bands
            hdi_prob: HDI probability (default 0.94)
            kind: 'line' (with shaded area) or 'dot' (with error bars)
            alpha: Transparency for shaded HDI area
        """
        category_col = self.category_col1 if category_col is None else category_col
        group_by = self.category_col2 if group_by is None else group_by

        def plot_func(ax):
            categories = self.df[category_col].unique()
            groups = self.df[group_by].unique()

            for cat in categories:
                means = []
                lower_bounds = []
                upper_bounds = []

                for grp in groups:
                    values = self.df[(self.df[category_col] == cat) & 
                                (self.df[group_by] == grp)][value_col].values

                    if len(values) > 0:
                        if stat == 'mean':
                            means.append(np.mean(values))
                        elif stat == 'median':
                            means.append(np.median(values))

                        if show_hdi and len(values) > 1:
                            interval = hdi(values, hdi_prob=hdi_prob)
                            lower_bounds.append(interval[0])
                            upper_bounds.append(interval[1])
                        else:
                            lower_bounds.append(means[-1])
                            upper_bounds.append(means[-1])
                    else:
                        means.append(np.nan)
                        lower_bounds.append(np.nan)
                        upper_bounds.append(np.nan)

                x = range(len(groups))

                if kind == 'line':
                    ax.plot(x, means, marker='o', label=cat)
                    if show_hdi:
                        ax.fill_between(x, lower_bounds, upper_bounds, alpha=alpha)
                elif kind == 'dot':
                    offset = np.linspace(-0.2, 0.2, len(categories))
                    cat_idx = list(categories).index(cat)  # Get current category index
                    x_offset = [xi + offset[cat_idx] for xi in x]

                    ax.errorbar(x_offset, means, 
                            yerr=[[m - l for m, l in zip(means, lower_bounds)],
                                    [u - m for m, u in zip(means, upper_bounds)]],
                            fmt='o', capsize=5, label=cat)

            ax.set_xticks(x)
            ax.set_xticklabels(groups)
            ax.legend()
            return ax

        title = f"{stat} of {value_col} by {category_col} across {group_by}" if title is None else title
        xlabel = group_by if xlabel is None else xlabel
        ylabel = f"{stat} {value_col}" if ylabel is None else ylabel

        return self._create_plot(plot_func, figsize, title, xlabel, ylabel, callbacks)

    def category_mean_by_group_faceted(self, value_col, category_col=None, group_by=None,
                                    facet_col=None, x_axis='group', x_order=None,
                                    stat='mean', show_hdi=True, hdi_prob=0.94,
                                    treat_as_continuous=False, ncols=3,
                                    title=None, xlabel=None, ylabel=None,
                                    alpha=0.3, figsize=None, callbacks=None, legend='right'):
        """Plot mean/median by category and group in faceted subplots.

        Args:
            value_col: Column containing values to aggregate
            category_col: First categorical column (e.g., 'agent')
            group_by: Second categorical column (e.g., 'topic')
            facet_col: Column to create separate subplots for
            x_axis: 'category' or 'group' - which goes on x-axis
            x_order: Custom order for x-axis categories (list), or None for alphabetical
            stat: 'mean' or 'median'
            show_hdi: Whether to show HDI bands/error bars
            treat_as_continuous: If True, connect points with lines
            ncols: Number of columns in facet grid
            legend: Legend position ('right', 'bottom', 'top', 'left', or None)
        """
        category_col = self.category_col1 if category_col is None else category_col
        group_by = self.category_col2 if group_by is None else group_by

        # Determine x-axis order once for all subplots
        x_col = category_col if x_axis == 'category' else group_by
        if x_order is not None:
            x_categories_ordered = x_order
        else:
            x_categories_ordered = sorted(self.df[x_col].unique())

        facet_values = self.df[facet_col].unique()
        data_subsets = [self.df[self.df[facet_col] == fv] for fv in facet_values]
        subplot_titles = list(facet_values)

        n_facets = len(facet_values)
        nrows = (n_facets + ncols - 1) // ncols

        def plot_func(ax, data):
            means, lower_bounds, upper_bounds = [], [], []

            for x_cat in x_categories_ordered:
                subset = data[data[x_col] == x_cat]
                if len(subset) > 0:
                    values = subset[value_col].values

                    if len(values) > 0:
                        means.append(np.mean(values) if stat == 'mean' else np.median(values))

                        if show_hdi and len(values) > 1:
                            interval = hdi(values, hdi_prob=hdi_prob)
                            lower_bounds.append(interval[0])
                            upper_bounds.append(interval[1])
                        else:
                            lower_bounds.append(means[-1])
                            upper_bounds.append(means[-1])
                    else:
                        means.append(np.nan)
                        lower_bounds.append(np.nan)
                        upper_bounds.append(np.nan)
                else:
                    means.append(np.nan)
                    lower_bounds.append(np.nan)
                    upper_bounds.append(np.nan)

            x = range(len(x_categories_ordered))

            if treat_as_continuous:
                ax.plot(x, means, marker='o')
                if show_hdi:
                    ax.fill_between(x, lower_bounds, upper_bounds, alpha=alpha)
            else:
                ax.errorbar(x, means,
                        yerr=[[m - l for m, l in zip(means, lower_bounds)],
                            [u - m for m, u in zip(means, upper_bounds)]],
                        fmt='o', capsize=5)

            ax.set_xticks(x)
            ax.set_xticklabels(x_categories_ordered, rotation=45, ha='right')
            return ax

        fig, axes = self._create_facet_plot(nrows, ncols, plot_func, data_subsets,
                                        figsize, subplot_titles, title, xlabel, ylabel, callbacks)

        # Handle legend (if needed in future for multi-line version)
        if legend is not None:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:  # Only add legend if there are items
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
            for ax in axes:
                legend_obj = ax.get_legend()
                if legend_obj:
                    legend_obj.remove()

        return fig, axes






