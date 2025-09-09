"""Feature selection module with PSI, IV filtering and correlation analysis"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .feature_engineer import FeatureEngineer


class FeatureSelector:
    """Handles feature selection including PSI, IV, correlation and forward selection"""
    
    def __init__(self, config):
        self.config = config
        self.engineer = FeatureEngineer(config)
        self.psi_results_ = {}
        self.iv_results_ = {}
        self.correlation_matrix_ = None
        
    def select_features(self, train: pd.DataFrame, test: Optional[pd.DataFrame] = None, 
                       oot: Optional[pd.DataFrame] = None) -> Dict:
        """Main feature selection pipeline"""
        
        print("Starting feature selection...")
        results = {'final_features': []}
        
        # Get initial feature list (exclude target, id, time cols)
        features = [col for col in train.columns 
                   if col not in [self.config.target_col, self.config.id_col, self.config.time_col]]
        
        # Step 1: Calculate IV for all features
        print("  1. Calculating Information Values...")
        self.iv_results_ = self.calculate_iv(train, features)
        
        # Step 2: Filter by minimum IV
        features = self.filter_by_iv(features, self.iv_results_)
        print(f"     After IV filter: {len(features)} features")
        
        # Step 3: Calculate PSI if test/oot provided
        if self.config.enable_psi and (test is not None or oot is not None):
            print("  2. Calculating PSI...")
            self.psi_results_ = self.calculate_psi(train, test, oot, features)
            features = self.filter_by_psi(features, self.psi_results_)
            print(f"     After PSI filter: {len(features)} features")
        
        # Step 4: Remove highly correlated features
        print("  3. Removing correlated features...")
        features = self.remove_correlated(train[features], features)
        print(f"     After correlation filter: {len(features)} features")
        
        # Step 5: Boruta selection if enabled
        if self.config.use_boruta and len(features) > 10:
            print("  4. Running Boruta selection...")
            X = train[features]
            y = train[self.config.target_col]
            features = self.engineer.boruta_selection(X, y)
            print(f"     After Boruta: {len(features)} features")
        
        # Step 6: Forward selection if enabled
        if self.config.forward_selection and len(features) > 5:
            print("  5. Running forward selection...")
            X = train[features]
            y = train[self.config.target_col]
            features = self.engineer.forward_selection(X, y, max_features=self.config.max_features)
            print(f"     After forward selection: {len(features)} features")
        
        # Step 7: Noise sentinel check if enabled
        if self.config.use_noise_sentinel:
            print("  6. Running noise sentinel check...")
            X = train[features]
            y = train[self.config.target_col]
            features = self.engineer.noise_sentinel_check(X, y, features)
        
        # Step 8: VIF check if enabled
        if self.config.vif_threshold and len(features) > 2:
            print("  7. Checking VIF...")
            vif_data = self.engineer.calculate_vif(train[features], self.config.vif_threshold)
            if not vif_data.empty:
                high_vif_features = vif_data[vif_data['VIF'] > self.config.vif_threshold]['feature'].tolist()
                features = [f for f in features if f not in high_vif_features]
                print(f"     After VIF filter: {len(features)} features")
        
        results['final_features'] = features
        results['iv_scores'] = self.iv_results_
        results['psi_scores'] = self.psi_results_
        
        print(f"\nFinal selected features: {len(features)}")
        return results
    
    def calculate_iv(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Calculate Information Value for features"""
        iv_list = []
        target = df[self.config.target_col]
        
        for feature in features:
            # For simplicity, use basic IV calculation
            # In production, this should use WOE bins
            try:
                # Group by feature and calculate event rates
                grouped = df.groupby(feature)[self.config.target_col].agg(['sum', 'count'])
                grouped['non_event'] = grouped['count'] - grouped['sum']
                grouped['event_rate'] = grouped['sum'] / grouped['sum'].sum()
                grouped['non_event_rate'] = grouped['non_event'] / grouped['non_event'].sum()
                
                # Calculate WOE and IV
                grouped['woe'] = np.log(grouped['event_rate'] / grouped['non_event_rate'] + 0.0001)
                grouped['iv'] = (grouped['event_rate'] - grouped['non_event_rate']) * grouped['woe']
                
                total_iv = grouped['iv'].sum()
                iv_list.append({'feature': feature, 'iv': total_iv})
            except Exception:
                iv_list.append({'feature': feature, 'iv': 0.0})
        
        return pd.DataFrame(iv_list).sort_values('iv', ascending=False)
    
    def filter_by_iv(self, features: List[str], iv_df: pd.DataFrame) -> List[str]:
        """Filter features by minimum IV threshold"""
        valid_features = iv_df[iv_df['iv'] >= self.config.iv_min]['feature'].tolist()
        return [f for f in features if f in valid_features]
    
    def calculate_psi(self, train: pd.DataFrame, test: Optional[pd.DataFrame], 
                     oot: Optional[pd.DataFrame], features: List[str]) -> Dict:
        """Calculate Population Stability Index"""
        psi_scores = {}
        
        for feature in features:
            psi_scores[feature] = {}
            
            if test is not None:
                psi_scores[feature]['test'] = self._calculate_feature_psi(
                    train[feature], test[feature]
                )
            
            if oot is not None:
                psi_scores[feature]['oot'] = self._calculate_feature_psi(
                    train[feature], oot[feature]
                )
        
        return psi_scores
    
    def _calculate_feature_psi(self, expected: pd.Series, actual: pd.Series, 
                               bins: int = 10) -> float:
        """Calculate PSI between two distributions"""
        try:
            # Create bins from expected distribution
            _, bin_edges = pd.qcut(expected.dropna(), q=bins, retbins=True, duplicates='drop')
            
            # Calculate distributions
            expected_dist = pd.cut(expected, bins=bin_edges, include_lowest=True).value_counts()
            actual_dist = pd.cut(actual, bins=bin_edges, include_lowest=True).value_counts()
            
            # Normalize
            expected_dist = expected_dist / len(expected)
            actual_dist = actual_dist / len(actual)
            
            # Calculate PSI
            psi = 0
            for bucket in expected_dist.index:
                e = expected_dist[bucket] if bucket in expected_dist.index else 0.0001
                a = actual_dist[bucket] if bucket in actual_dist.index else 0.0001
                psi += (a - e) * np.log(a / e)
            
            return float(psi)
        except Exception:
            return 0.0
    
    def calculate_score_psi(self, train_scores: np.ndarray, test_scores: np.ndarray, 
                           bins: int = 10) -> Tuple[float, pd.DataFrame]:
        """Calculate PSI for model scores with segment details"""
        
        # Create score segments
        _, bin_edges = pd.qcut(train_scores, q=bins, retbins=True, duplicates='drop')
        
        train_dist = pd.cut(train_scores, bins=bin_edges, include_lowest=True).value_counts()
        test_dist = pd.cut(test_scores, bins=bin_edges, include_lowest=True).value_counts()
        
        # Normalize
        train_pct = train_dist / len(train_scores)
        test_pct = test_dist / len(test_scores)
        
        # Create segment report
        segments = []
        total_psi = 0
        
        for bucket in train_pct.index:
            train_p = train_pct[bucket] if bucket in train_pct.index else 0.0001
            test_p = test_pct[bucket] if bucket in test_pct.index else 0.0001
            
            psi_contrib = (test_p - train_p) * np.log(test_p / train_p)
            total_psi += psi_contrib
            
            segments.append({
                'segment': str(bucket),
                'train_pct': train_p * 100,
                'test_pct': test_p * 100,
                'diff_pct': (test_p - train_p) * 100,
                'psi_contribution': psi_contrib
            })
        
        segment_df = pd.DataFrame(segments)
        return total_psi, segment_df
    
    def filter_by_psi(self, features: List[str], psi_scores: Dict) -> List[str]:
        """Filter features by PSI threshold"""
        keep_features = []
        
        for feature in features:
            if feature in psi_scores:
                # Check test PSI
                test_psi = psi_scores[feature].get('test', 0)
                oot_psi = psi_scores[feature].get('oot', 0)
                
                max_psi = max(test_psi, oot_psi)
                if max_psi < self.config.psi_threshold:
                    keep_features.append(feature)
            else:
                keep_features.append(feature)
        
        return keep_features
    
    def remove_correlated(self, df: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove highly correlated features"""
        
        # Calculate correlation matrix
        corr_matrix = df.corr(method='spearman')
        self.correlation_matrix_ = corr_matrix
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = set()
        for column in upper_triangle.columns:
            if column in to_drop:
                continue
            
            # Find correlated features
            correlated = upper_triangle[column][
                abs(upper_triangle[column]) > self.config.rho_threshold
            ].index.tolist()
            
            if correlated:
                # Keep the one with highest IV
                all_features = [column] + correlated
                iv_scores = self.iv_results_.set_index('feature')['iv']
                
                # Get IV scores for these features
                feature_ivs = {}
                for f in all_features:
                    if f in iv_scores.index:
                        feature_ivs[f] = iv_scores[f]
                    else:
                        feature_ivs[f] = 0
                
                # Keep feature with highest IV
                best_feature = max(feature_ivs.items(), key=lambda x: x[1])[0]
                for f in all_features:
                    if f != best_feature:
                        to_drop.add(f)
        
        return [f for f in features if f not in to_drop]