"""
Visualization utilities for model evaluation and reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


class VisualizationHelper:
    """Helper class for creating model evaluation visualizations"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), style: str = 'seaborn'):
        """
        Initialize visualization helper
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size
        style : str
            Matplotlib style
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(style)
        
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: Union[np.ndarray, Dict[str, np.ndarray]],
        labels: Optional[List[str]] = None,
        title: str = 'ROC Curve',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot ROC curve(s)
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_scores : np.ndarray or dict
            Predicted scores. If dict, multiple curves will be plotted
        labels : List[str], optional
            Labels for multiple curves
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            The figure object
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert single array to dict for uniform handling
        if isinstance(y_scores, np.ndarray):
            y_scores = {'Model': y_scores}
            if labels is None:
                labels = ['Model']
        elif labels is None:
            labels = list(y_scores.keys())
        
        # Plot ROC curves
        for label, scores in y_scores.items():
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})', linewidth=2)
        
        # Plot random classifier
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', linewidth=1)
        
        # Formatting
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: Union[np.ndarray, Dict[str, np.ndarray]],
        labels: Optional[List[str]] = None,
        title: str = 'Precision-Recall Curve',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve(s)
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_scores : np.ndarray or dict
            Predicted scores
        labels : List[str], optional
            Labels for multiple curves
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            The figure object
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert single array to dict for uniform handling
        if isinstance(y_scores, np.ndarray):
            y_scores = {'Model': y_scores}
            if labels is None:
                labels = ['Model']
        elif labels is None:
            labels = list(y_scores.keys())
        
        # Calculate baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        
        # Plot PR curves
        for label, scores in y_scores.items():
            precision, recall, _ = precision_recall_curve(y_true, scores)
            avg_precision = np.mean(precision)
            ax.plot(recall, precision, label=f'{label} (AP = {avg_precision:.3f})', linewidth=2)
        
        # Plot baseline
        ax.plot([0, 1], [baseline, baseline], 'k--', 
                label=f'Baseline (AP = {baseline:.3f})', linewidth=1)
        
        # Formatting
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_scores: Union[np.ndarray, Dict[str, np.ndarray]],
        n_bins: int = 10,
        labels: Optional[List[str]] = None,
        title: str = 'Calibration Plot',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot calibration curve(s)
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_scores : np.ndarray or dict
            Predicted probabilities
        n_bins : int
            Number of bins for calibration
        labels : List[str], optional
            Labels for multiple curves
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            The figure object
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert single array to dict for uniform handling
        if isinstance(y_scores, np.ndarray):
            y_scores = {'Model': y_scores}
            if labels is None:
                labels = ['Model']
        elif labels is None:
            labels = list(y_scores.keys())
        
        # Plot calibration curves
        for label, scores in y_scores.items():
            fraction_pos, mean_pred = calibration_curve(y_true, scores, n_bins=n_bins)
            ax.plot(mean_pred, fraction_pos, marker='o', label=label, linewidth=2, markersize=8)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1)
        
        # Formatting
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_score_distribution(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        n_bins: int = 30,
        title: str = 'Score Distribution',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot score distribution by class
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_scores : np.ndarray
            Predicted scores
        n_bins : int
            Number of bins for histogram
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            The figure object
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Separate scores by class
        scores_pos = y_scores[y_true == 1]
        scores_neg = y_scores[y_true == 0]
        
        # Plot histograms
        ax.hist(scores_neg, bins=n_bins, alpha=0.5, label='Negative class', 
                color='blue', density=True)
        ax.hist(scores_pos, bins=n_bins, alpha=0.5, label='Positive class', 
                color='red', density=True)
        
        # Add vertical line for mean scores
        ax.axvline(np.mean(scores_neg), color='blue', linestyle='--', 
                  linewidth=2, label=f'Mean negative: {np.mean(scores_neg):.3f}')
        ax.axvline(np.mean(scores_pos), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean positive: {np.mean(scores_pos):.3f}')
        
        # Formatting
        ax.set_xlabel('Predicted Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        normalize: bool = False,
        title: str = 'Confusion Matrix',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        labels : List[str], optional
            Class labels
        normalize : bool
            Whether to normalize the matrix
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            The figure object
        """
        
        from sklearn.metrics import confusion_matrix
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        # Plot heatmap
        if labels is None:
            labels = ['Negative', 'Positive']
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        # Formatting
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        top_n: int = 20,
        title: str = 'Feature Importance',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot feature importance
        
        Parameters:
        -----------
        feature_names : List[str]
            Feature names
        importance_values : np.ndarray
            Importance values
        top_n : int
            Number of top features to show
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            The figure object
        """
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create dataframe and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(importance_df))
        ax.barh(y_pos, importance_df['importance'].values, color='skyblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['feature'].values)
        ax.invert_yaxis()
        
        # Formatting
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_lift_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        n_bins: int = 10,
        title: str = 'Lift Curve',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot lift curve
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_scores : np.ndarray
            Predicted scores
        n_bins : int
            Number of bins
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        plt.Figure
            The figure object
        """
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1]//2))
        
        # Create dataframe
        df = pd.DataFrame({'y_true': y_true, 'y_score': y_scores})
        df = df.sort_values('y_score', ascending=False)
        
        # Calculate lift for each bin
        df['bin'] = pd.qcut(df.index, n_bins, labels=False, duplicates='drop')
        
        lift_data = []
        cumulative_data = []
        base_rate = y_true.mean()
        
        for i in range(n_bins):
            bin_data = df[df['bin'] <= i]
            bin_rate = bin_data['y_true'].mean()
            lift = bin_rate / base_rate if base_rate > 0 else 0
            
            lift_data.append(lift)
            
            # Cumulative
            cum_positives = bin_data['y_true'].sum()
            cum_total = len(bin_data)
            cum_rate = cum_positives / cum_total if cum_total > 0 else 0
            cum_lift = cum_rate / base_rate if base_rate > 0 else 0
            cumulative_data.append(cum_lift)
        
        # Plot lift curve
        x = np.arange(1, n_bins + 1) * (100 / n_bins)
        ax1.plot(x, lift_data, marker='o', linewidth=2, markersize=8, color='blue')
        ax1.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax1.set_xlabel('Percentile', fontsize=12)
        ax1.set_ylabel('Lift', fontsize=12)
        ax1.set_title('Lift by Percentile', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot cumulative lift
        ax2.plot(x, cumulative_data, marker='o', linewidth=2, markersize=8, color='green')
        ax2.axhline(y=1, color='r', linestyle='--', label='Baseline')
        ax2.set_xlabel('Percentile', fontsize=12)
        ax2.set_ylabel('Cumulative Lift', fontsize=12)
        ax2.set_title('Cumulative Lift', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_model_evaluation_report(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        save_dir: Optional[str] = None
    ) -> Dict[str, plt.Figure]:
        """
        Create comprehensive model evaluation report with multiple plots
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_scores : np.ndarray
            Predicted scores
        y_pred : np.ndarray, optional
            Binary predictions
        threshold : float
            Threshold for binary classification
        save_dir : str, optional
            Directory to save plots
            
        Returns:
        --------
        Dict[str, plt.Figure]
            Dictionary of figure objects
        """
        
        if y_pred is None:
            y_pred = (y_scores >= threshold).astype(int)
        
        figures = {}
        
        # ROC Curve
        figures['roc'] = self.plot_roc_curve(
            y_true, y_scores,
            save_path=f"{save_dir}/roc_curve.png" if save_dir else None
        )
        
        # Precision-Recall Curve
        figures['pr'] = self.plot_precision_recall_curve(
            y_true, y_scores,
            save_path=f"{save_dir}/pr_curve.png" if save_dir else None
        )
        
        # Calibration Curve
        figures['calibration'] = self.plot_calibration_curve(
            y_true, y_scores,
            save_path=f"{save_dir}/calibration_curve.png" if save_dir else None
        )
        
        # Score Distribution
        figures['distribution'] = self.plot_score_distribution(
            y_true, y_scores,
            save_path=f"{save_dir}/score_distribution.png" if save_dir else None
        )
        
        # Confusion Matrix
        figures['confusion'] = self.plot_confusion_matrix(
            y_true, y_pred,
            save_path=f"{save_dir}/confusion_matrix.png" if save_dir else None
        )
        
        # Lift Curve
        figures['lift'] = self.plot_lift_curve(
            y_true, y_scores,
            save_path=f"{save_dir}/lift_curve.png" if save_dir else None
        )
        
        return figures