import numpy as np
import pandas as pd
from scipy.stats import binomtest
from typing import Dict, List, Optional, Union

def _validate_dataframe(df: pd.DataFrame, required_cols: List[str], context: str = "") -> None:
    """Validate that DataFrame contains required columns."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{context}: Missing required columns: {missing}. "
                        f"Available columns: {list(df.columns)}")

def classify_volatility_regime(sigma_annual: float) -> str:
    """
    Classify market volatility regime based on annualized volatility.
    
    Args:
        sigma_annual: Annualized volatility as percentage (e.g., 20.0 for 20%)
    
    Returns:
        String: 'Low' (<15%), 'Medium' (15-30%), or 'High' (>30%)
    """
    if sigma_annual < 15:
        return 'Low'
    elif sigma_annual < 30:
        return 'Medium'
    else:
        return 'High'

def calculate_directional_accuracy(df_results: pd.DataFrame, 
                                   num_days: Optional[int] = None,
                                   mu_col: str = 'mu') -> Dict[str, Union[float, int]]:
    """
    Calculate percentage of correct up/down predictions.
    
    Uses the drift (mu) to calculate predicted median price if available,
    otherwise falls back to using the midpoint of confidence intervals.
    
    Args:
        df_results: DataFrame with 'Starting Price', 'Actual Price', and bounds
        num_days: Forecast horizon in days (required if using mu_col)
        mu_col: Column name for drift parameter (if available)
    
    Returns:
        dict with directional accuracy metrics
    """
    df = df_results.copy()
    
    # Check required columns
    _validate_dataframe(df, ['Starting Price', 'Actual Price', 'Upper Bound', 'Lower Bound'], 
                       "calculate_directional_accuracy")
    
    # Calculate predicted direction
    if mu_col in df.columns and num_days is not None:
        # Use proper GBM median: S0 * exp(mu * T)
        predicted_price = df['Starting Price'] * np.exp(df[mu_col] * num_days)
        df['predicted_direction'] = np.sign(predicted_price - df['Starting Price'])
    else:
        # Fallback: use midpoint of confidence interval
        midpoint = (df['Upper Bound'] + df['Lower Bound']) / 2
        df['predicted_direction'] = np.sign(midpoint - df['Starting Price'])
    
    # Actual direction
    df['actual_direction'] = np.sign(df['Actual Price'] - df['Starting Price'])
    
    # Calculate metrics
    correct = (df['predicted_direction'] == df['actual_direction']).sum()
    total = len(df)
    
    upward_correct = ((df['predicted_direction'] > 0) & (df['actual_direction'] > 0)).sum()
    upward_total = (df['actual_direction'] > 0).sum()
    
    downward_correct = ((df['predicted_direction'] < 0) & (df['actual_direction'] < 0)).sum()
    downward_total = (df['actual_direction'] < 0).sum()
    
    return {
        'overall_accuracy': correct / total if total > 0 else 0,
        'upward_accuracy': upward_correct / upward_total if upward_total > 0 else 0,
        'downward_accuracy': downward_correct / downward_total if downward_total > 0 else 0,
        'total_tests': total,
        'method': 'gbm_median' if mu_col in df.columns else 'midpoint_fallback'
    }

def calculate_calibration_metrics(df_results: pd.DataFrame, 
                                  confidence_level: float = 0.9) -> Dict:
    """
    Calculate comprehensive calibration metrics for backtesting results.
    
    Args:
        df_results: DataFrame with validation results including 'Within Bounds', 
                   'Lower Bound', 'Upper Bound', 'Starting Price', 'Actual Price',
                   and optionally 'mu' and 'Volatility'
        confidence_level: Expected confidence level (default 0.9 for 90%)
    
    Returns:
        dict with calibration statistics including hit rate, regime analysis, 
        worst misses, and statistical significance tests
    """
    # Filter valid results
    df = df_results[df_results['Actual Price'].notna()].copy()
    
    if len(df) == 0:
        return {
            'overall_hit_rate': 0,
            'regime_analysis': {},
            'worst_misses': [],
            'margin_of_error': 0,
            'p_value': 1.0,
            'num_hits': 0,
            'confidence_level': confidence_level,
            'total_tests': 0,
            'error': 'No valid results with actual prices'
        }
    
    # Validate required columns exist
    _validate_dataframe(df, ['Within Bounds', 'Lower Bound', 'Upper Bound', 
                            'Starting Price', 'Actual Price'], 
                       "calculate_calibration_metrics")
    
    # Basic hit rate
    hit_rate = (df['Within Bounds'].sum() / len(df)) * 100
    
    # Regime-based analysis (if volatility available)
    regime_stats = {}
    if 'Volatility' in df.columns:
        df['Volatility Regime'] = df['Volatility'].apply(classify_volatility_regime)
        
        for regime in ['Low', 'Medium', 'High']:
            regime_df = df[df['Volatility Regime'] == regime]
            if len(regime_df) > 0:
                regime_hit_rate = (regime_df['Within Bounds'].sum() / len(regime_df)) * 100
                
                # Calculate average miss magnitude
                misses = regime_df[~regime_df['Within Bounds']]
                if len(misses) > 0:
                    miss_magnitude = abs(
                        (misses['Actual Price'] - misses['Starting Price']) / misses['Starting Price']
                    ).mean() * 100
                else:
                    miss_magnitude = 0.0
                
                regime_stats[regime] = {
                    'hit_rate': regime_hit_rate,
                    'count': len(regime_df),
                    'avg_miss_magnitude': miss_magnitude
                }
    
    # Distance from bounds analysis
    df['Distance to Lower'] = (df['Actual Price'] - df['Lower Bound']) / df['Starting Price'] * 100
    df['Distance to Upper'] = (df['Upper Bound'] - df['Actual Price']) / df['Starting Price'] * 100
    
    # Identify failure cases (worst misses)
    df['Miss Magnitude'] = 0.0
    misses = df[~df['Within Bounds']].copy()
    
    if len(misses) > 0:
        below_lower = misses[misses['Actual Price'] < misses['Lower Bound']]
        above_upper = misses[misses['Actual Price'] > misses['Upper Bound']]
        
        if len(below_lower) > 0:
            df.loc[below_lower.index, 'Miss Magnitude'] = (
                (below_lower['Lower Bound'] - below_lower['Actual Price']) / below_lower['Starting Price'] * 100
            )
        
        if len(above_upper) > 0:
            df.loc[above_upper.index, 'Miss Magnitude'] = (
                (above_upper['Actual Price'] - above_upper['Upper Bound']) / above_upper['Starting Price'] * 100
            )
    

    worst_misses_cols = ['Actual Price', 'Lower Bound', 'Upper Bound', 'Miss Magnitude']
    if 'Test Date' in df.columns:
        worst_misses_cols.insert(0, 'Test Date') 
    if 'Volatility' in df.columns:
        worst_misses_cols.append('Volatility')

    worst_misses = df.nlargest(min(5, len(df)), 'Miss Magnitude')[worst_misses_cols].to_dict('records') if len(df) > 0 else []
    
    # Statistical significance
    n = len(df)
    num_hits = df['Within Bounds'].sum()
    margin_of_error = 1.96 * np.sqrt(confidence_level * (1 - confidence_level) / n) * 100 if n > 0 else 0
    p_value = binomtest(int(num_hits), n, confidence_level, alternative='two-sided').pvalue if n > 0 else 1.0
    
    return {
        'overall_hit_rate': hit_rate,
        'regime_analysis': regime_stats,
        'worst_misses': worst_misses,
        'margin_of_error': margin_of_error,
        'p_value': p_value,
        'total_tests': n,
        'num_hits': int(num_hits),
        'confidence_level': confidence_level
    }

def identify_failure_patterns(df_results: pd.DataFrame, 
                             threshold: float = 0.15) -> Dict:
    """
    Identify common patterns in validation failures.
    
    Args:
        df_results: DataFrame with validation results
        threshold: Miss threshold as decimal (0.15 = 15% miss)
    
    Returns:
        dict with failure pattern analysis
    """
    _validate_dataframe(df_results, ['Within Bounds', 'Actual Price', 'Lower Bound', 
                                    'Upper Bound', 'Starting Price'], 
                       "identify_failure_patterns")
    
    misses = df_results[~df_results['Within Bounds']].copy()
    
    if len(misses) == 0:
        return {'major_failures': 0, 'total_misses': 0, 'patterns': [], 'failure_rate': 0}
    
    # Calculate miss severity
    misses['Miss Magnitude'] = 0.0
    below_lower = misses[misses['Actual Price'] < misses['Lower Bound']]
    above_upper = misses[misses['Actual Price'] > misses['Upper Bound']]
    
    if len(below_lower) > 0:
        misses.loc[below_lower.index, 'Miss Magnitude'] = (
            (below_lower['Lower Bound'] - below_lower['Actual Price']) / below_lower['Starting Price']
        ).abs()
    
    if len(above_upper) > 0:
        misses.loc[above_upper.index, 'Miss Magnitude'] = (
            (above_upper['Actual Price'] - above_upper['Upper Bound']) / above_upper['Starting Price']
        ).abs()
    
    major_failures = misses[misses['Miss Magnitude'] > threshold]
    
    patterns = []
    if len(major_failures) > 0:
        if 'Volatility' in major_failures.columns:
            avg_vol = major_failures['Volatility'].mean()
            patterns.append(f"Major failures occur at {avg_vol:.1f}% avg volatility")
        
        downside_skew = (major_failures['Actual Price'] < major_failures['Lower Bound']).sum()
        if downside_skew > len(major_failures) / 2:
            patterns.append("Failures skewed toward downside (crashes)")
        elif downside_skew < len(major_failures) / 2:
            patterns.append("Failures skewed toward upside (unexpected rallies)")
    
    return {
        'major_failures': len(major_failures),
        'total_misses': len(misses),
        'patterns': patterns,
        'failure_rate': len(major_failures) / len(df_results) * 100 if len(df_results) > 0 else 0,
        'avg_miss_magnitude': misses['Miss Magnitude'].mean() if len(misses) > 0 else 0
    }