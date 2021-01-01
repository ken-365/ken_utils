import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

### Visualization
class eda_plot:
    def __init__(self):
        pass
    def plot_series(series, n_steps, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$", ):
        '''
        Create plot from time-series data
        Input:
            n_steps = time_length to show
        Output: plot
        '''
        plt.plot(series, ".-")
        if y is not None:
            plt.plot(n_steps, y, "bx", markersize=10)
        if y_pred is not None:
            plt.plot(n_steps, y_pred, "ro")
        plt.grid(True)
        if x_label:
            plt.xlabel(x_label, fontsize=16)
        if y_label:
            plt.ylabel(y_label, fontsize=16, rotation=0)
        plt.hlines(0, 0, 100, linewidth=1)
        plt.axis([0, n_steps + 1, -1, 1])

    def eda_by_discrete_feature(feature,target,hue,df):
        '''
        Description: Plot discrete features
        Input: 
            1. a features on x axis
            2. target variable interested on y axis
            3. hue devide data by (eg men/ women)
            4. dataframe
        Output: plot
        Implementation :
            eda_by_discrete_feature('country','conversion','sex',df)

        '''
        grouped = df[[feature, target]].groupby(feature).mean().reset_index()
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(28, 9))
        sns.countplot(x=feature, hue=hue, data=df, ax=ax[0])
        ax[0].set_title('Count Plot of '+feature, fontsize=16)
        ax[0].set_yscale('log')
        sns.barplot(x=feature, y=target, data=df, ax=ax[1]);
        ax[1].set_title('Mean Conversion Rate by '+feature, fontsize=16)
        plt.tight_layout()
        plt.show()


    def eda_by_continuous_feature(feature,target,data):
        '''
        Description: Plot continuous features
        Input: list of features columns name from dataframe
        Output: plot
        Implementation :
            for feature in ['age', 'total_pages_visited']: #feature names
            eda_by_discrete_feature(feature)
        '''
        grouped = data[[feature, target]].groupby(feature).mean().reset_index()
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
        sns.distplot(data[data[target] == 0][feature], 
                     label='churn 0', ax=ax[0])
        sns.distplot(data[data[target] == 1][feature], 
                     label='churn 1', ax=ax[0])
        ax[0].set_title('Count Plot of '+feature, fontsize=16)
        ax[0].legend()
        ax[1].plot(grouped[feature], grouped[target], '.-')
        ax[1].set_title('Mean churn week vs. '+feature, fontsize=16)
        ax[1].set_xlabel(feature)
        ax[1].set_ylabel('Mean churn week')
        ax[1].grid(True)
        plt.show()
        
    def add_text_barplot(ax, decimals=4, rot=30,fontsize=12):
        '''
        add text to barplot
        '''
        for p in ax.patches:
            ax.annotate(np.round(p.get_height(), decimals=decimals), 
                    (p.get_x()+p.get_width()/2.,p.get_height()), 
                    ha='center', 
                    va='bottom', 
                    xytext=(0, 10), 
                    rotation=rot,
                    fontsize=fontsize,
                    textcoords='offset points')

    def plot_categorical(df,feature,target_feature,figsize=(8,6),ylim2=None,rot=30,fontsize=12):
        '''
        create count, propotion, mean, total number bar plot for categorical features
        
        Implementation:
            cols_cat = ['subscription_monthly_cost','country','source']

            for cat in cols_cat:

                print(f'Feature: {cat}')
                display(sub.groupby(cat).agg({'is_active':'mean'}))
                plot_categorical(sub,cat,'is_active',figsize=(20,13),rot=90)
                print('='*100)
        
        '''
        fig, ax = plt.subplots(2,2,figsize=figsize)
        pd.crosstab(df[feature],df[target_feature]).plot.bar(ax=ax[0,0])
        pd.crosstab(df[feature],df[target_feature],normalize='index').plot.bar(ax=ax[0,1])
        grouped = df[[feature, target_feature]].groupby(feature).mean().reset_index()

        if ylim2:
            ax[0,1].set_ylim(0,ylim2)

        plt.suptitle(f'{feature} vs {target_feature}',fontsize=18,color='#d11d53',y=1.02)

        sns.countplot(x=feature,data=df,ax=ax[1,0])
        sns.barplot(x=feature, y=target_feature, data=df, ax=ax[1,1]);

        ax[0,0].set_title(f'Number of user by {feature} that {target_feature}')
        ax[0,1].set_title(f'Propotion of user by {feature} that {target_feature}')
        ax[1,0].set_title(f'Number of user by '+feature)
        ax[1,1].set_title(f'Mean {target_feature} Rate by {feature}')

        for i in range(2):
            for j in range(2):
                add_text_barplot(ax[i,j],rot=rot,fontsize=fontsize)
                ax[i,j].margins(0.18)

        plt.tight_layout()
        plt.show()
        
# neural net


# others
def jupyter_hidecode():
    print("""
    from IPython.display import HTML
# add button to show/ hide code
HTML('''<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
    """)