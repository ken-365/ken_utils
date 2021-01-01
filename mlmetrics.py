import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import sklearn.metrics as metrics
from sklearn.metrics import precision_score, recall_score,f1_score, accuracy_score ,roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.utils import resample
from itertools import cycle

#### Regression
class Regression:   
    def __init__(self):
        pass 
    def evaluate(y_true, y_pred,estimator):
        '''
        Return ind, value of many regression metrics in loop    
        then you need to  create data frame your self with code below

        ind, val = evaluate(ytrain, y_pred,lin_r)
        pd.DataFrame([val] ,index =[ ind] ,columns=['explained_variance ','r2 ','MAE ','MSE ','RMSE '])
        
        or  to compare multiple model
        
        ind, val = [],[]
        for estimator in [lin_r ,las,elas,ridg,ada ,extra ,gra ,rnd ] :  
            estimator.fit(Xtrain,ytrain)
            y_pred = estimator.predict(Xtrain)
            tmp1, tmp2  = evaluate(ytrain, y_pred,estimator)
            ind.append(tmp1)
            val.append(tmp2)
        result1 = pd.DataFrame(np.array(val),index = [ind ],columns=['explained_variance','r2','MAE','MSE','RMSE'])
        result1.sort_values(by=['MAE','RMSE','MSE'])
        
        '''
        # Regression metrics
        explained_variance=metrics.explained_variance_score(y_true, y_pred)
        mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
        mse=metrics.mean_squared_error(y_true, y_pred) 
        median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
        r2=metrics.r2_score(y_true, y_pred)

        return type(estimator).__name__,[round(explained_variance,6),round(r2,6),round(mean_absolute_error,6),round(mse,6),round(np.sqrt(mse),6)]

### Classification
class Classification:
    def __init__(self):
        pass 
    def evaluate(y_true, y_pred,estimator):
        '''
        Return ind, value of many regression metrics in loop    
        then you need to  create data frame your self with code below

        ind, val = evaluate(ytrain, y_pred,lin_r)
        pd.DataFrame([val] ,index =[ ind] ,columns=['explained_variance ','r2 ','MAE ','MSE ','RMSE '])
        
        or  to compare multiple model
        
        ind, val = [],[]
        for estimator in [lin_r ,las,elas,ridg,ada ,extra ,gra ,rnd ] :  
            estimator.fit(Xtrain,ytrain)
            y_pred = estimator.predict(Xtrain)
            tmp1, tmp2  = evaluate(ytrain, y_pred,estimator)
            ind.append(tmp1)
            val.append(tmp2)
        result1 = pd.DataFrame(np.array(val),index = [ind ],columns=['accuracy','log_loss,'MAE','MSE','RMSE'])
        result1.sort_values(by=['MAE','RMSE','MSE'])
        
        '''
        # Regression metrics
        accuracy=metrics.accuracy(y_true, y_pred)
        log_loss=metrics.log_loss(y_true, y_pred) 
        mse=metrics.mean_squared_error(y_true, y_pred) 
        median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
        r2=metrics.r2_score(y_true, y_pred)

        return type(estimator).__name__,[round(accuracy,6),round(r2,6),round(log_loss,6),round(mse,6),round(np.sqrt(mse),6)]

        
    def plot_confusion_matrix(y_true,y_pred,title , labels, normalize = True):
        '''
        Description: Plot confusion matrix
        Input: y_true,y_predict of shape (n_samples, n_outputs) , title of plot , list of label = ['Not-Churn','Churn'], want result as percent? for nomalize
        Output: plot
        Implementation : plot_confusion_matrix_2(y_true,y_pred, Is_churn_7d , ['Not-Churn','Churn'], normalize = True)
        '''
        mat = confusion_matrix(y_true,y_pred)
        if normalize :
            mat = mat/ mat.sum()
        labels= labels
        sns.heatmap(mat, square=True, annot=True, cbar=False,
                    cmap=plt.cm.Blues, # set theme
                    fmt='g', # set no truncate number
                    xticklabels=labels,
                    yticklabels=labels)
        plt.title(title + '\nAccuracy:{0:.3f} , Sample size = {1}'.format(accuracy_score(np.array(y_train),np.array(y_pred)),mat.sum()), fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Value')
        plt.ylabel('Actual');

    def plot_confusion_matrix_multilabel(y_true,y_predict,titles , labels, normalize = True):
        '''
        Description: Plot confusion matrix
        Input: y_true,y_predict of shape (n_samples, n_outputs) , title of plot , list of label = ['Not-Churn','Churn'], want result as percent? for nomalize
        Output: plot
        Implementation : plot_confusion_matrix_2(y_true,y_pred, ['Is_churn_7d','Is_churn_30d'] , ['Not-Churn','Churn'], normalize = True)
        '''
        mat = multilabel_confusion_matrix(np.array(y_true),np.array(y_predict))
        title = titles
        for i in range(len(title)):
            print('--'*25)
            print('\033[1m' + title[i] + '\033[0m' + '\nAccuracy:{0:.3f} , Sample size = {1}'.format(accuracy_score(np.array(y_true),np.array(y_predict)),np.array(y_true).shape[0]))       
            labels = labels
            submat = mat
            if normalize  :
                submat = mat[i]/ mat[i].sum()
            else :
                submat = mat[i]
            labels= labels
            sns.heatmap(submat, square=True, annot=True, cbar=False,
                        cmap=plt.cm.Blues, # set theme
                        fmt='g', # set no truncate number
                        xticklabels=labels,
                        yticklabels=labels)
            plt.xlabel('Predicted Value')
            plt.ylabel('Actual')
            plt.show()

    def plot_feature_importance(importance,names,model_type):
        """
        Description: Plot feature importances
        Input: list of features name, coef or feature importances , 'Name of model'
        Output: plot
        Implementation : plot_feature_importance(coef,features_name, 'LOGISTIC REGRESSION ')
        """
        #Create arrays from feature importance and feature names
        feature_importance = np.array(importance)
        feature_names = np.array(names)

        #Create a DataFrame using a Dictionary
        data={'feature_names':feature_names,'feature_importance':feature_importance}
        fi_df = pd.DataFrame(data)

        #Sort the DataFrame in order decreasing feature importance
        fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

        #Define size of bar plot
        plt.figure(figsize=(20,13))
        #Plot Searborn bar chart
        sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
        #Add chart labels
        plt.title(model_type + 'FEATURE IMPORTANCE')
        plt.xlabel('FEATURE IMPORTANCE')
        plt.ylabel('FEATURE NAMES')
    
    def plot_roc_auc_multilabel(y_true,y_pred,n_classes,line_width=2):
        """
        Description: Plot ROC_AUC_Curve for multi label classification
        Input: y_true, y_predict, number of unique multiclasses ,line_thickness in plot
        Output: plot
        Implementation :plot_roc_auc_multilabel(y_true,y_pred, 2,2)
        """
        # Compute micro-average ROC curve and ROC area
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = n_classes
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(np.array(y_true)[:,i], np.array(y_pred)[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(np.array(y_true).ravel(), np.array(y_pred).ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        lw = line_width
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure(figsize=(12,8))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic to multi-label')
        plt.legend(loc="lower right")
        plt.show()
# Deep learning
class Neural_net_eval:
    def __init__(self):
        pass
    def plot_trainval_compare(history,metric_train,metric_val):
        '''
        Description: compare eg loss or accuracy on training set and validation set by epochs
        Input: (history,'loss','val_loss'):
        Output: plot
        Implementation :
                plot_trainval_compare(history,'loss','val_loss'):
        '''
        history_dict = history.history
        loss_values = history_dict[metric_train]
        val_loss_values = history_dict[metric_val]
        print('average train '+metric_train,np.array(loss_values).mean())
        print('average validation '+metric_val,np.array(val_loss_values).mean())
        plt.plot(np.arange(len(loss_values)) + 0.5, loss_values, "b.-", label='train '+metric_train)
        plt.plot(np.arange(len(val_loss_values)) + 1, val_loss_values, "r.-", label='validation '+metric_val)
        plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        plt.title('Training and validation metric by epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric_train)
        plt.legend(fontsize=13.5)
        plt.show()

        
# experimentation
class experiment:
    def __init__(self):
        pass
    def run_ttest(df,target):
        '''
        Description: Run ttest
        Input: dataframe
        Output: plot
        Implementation : run_ttest(df).to_frame('Value').reset_index()
        or df.groupby('device').apply(run_ttest)
        '''
        test_group = df[df["test"] == 1][target]
        control_group = df[df["test"] == 0][target]
        test_mean = test_group.mean()
        control_mean = control_group.mean()

        result = stats.ttest_ind(control_group, test_group)
        conclusion = "Significant" if result.pvalue < 0.05 else "Not-Significant"
        is_significant = 1 if result.pvalue < 0.05 else 0

        return pd.Series({"number of test": test_group.shape[0],
                          "number of control": control_group.shape[0],
                          "mean churn (test)": test_mean,
                          "mean churn (control)": control_mean,
                          "P-value": result.pvalue,
                          "Conclusion": conclusion,
                          "is_significant": is_significant
                         })
    
        def calcualte_sample_size(effect_size, alpha,sample_size):
            '''
            Description: calcualte sample size
            Input: effect_size, alpha,sample_size
            Output: plot
            Implementation : calcualte_sample_size(0.05,0.05,5000)
            '''
            X = []
            y = []
            for ss in range(int(sample_size/5),int(sample_size*2)):
                power = statsmodels.stats.power.tt_ind_solve_power(effect_size=effect_size,alpha=alpha,nobs1=ss)
                X.append(ss)
                y.append(power)
                if power > 0.8:
                    n ,p = ss,power
            print('The sample size require to have power of 0.8 given the parameter is',statsmodels.stats.power.tt_ind_solve_power(alpha=alpha,effect_size=effect_size,power=0.8),'\n')
            plt.figure(figsize=(8, 6)) 
            plt.title('Power for each sample size')
            plt.xlabel('Sample size')
            plt.ylabel('Power')
            sns.lineplot(x=X,y=y)