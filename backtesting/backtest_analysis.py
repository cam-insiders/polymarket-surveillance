"""
Backtest analysis and visualization.
Compares results across different parameter configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import logging
from pathlib import Path


class BacktestAnalyzer:
    """
    Analyzes and visualizes backtest results.
    """
    
    def __init__(self, results_dir: str = "backtest_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def compare_configurations(self, config_results: Dict[str, List]) -> pd.DataFrame:
        """
        Compare backtest results across different parameter configurations.
        """
        comparison = []
        
        for config_name, results in config_results.items():
            total_trades = sum(r.total_trades for r in results)
            total_alerts = sum(r.alerts_generated for r in results)
            
            # Aggregate detector fire counts
            detector_fires = {}
            for result in results:
                for detector, count in result.detector_stats.items():
                    detector_fires[detector] = detector_fires.get(detector, 0) + count
            
            # Calculate metrics
            alert_rate = total_alerts / total_trades if total_trades > 0 else 0
            
            comparison.append({
                'config': config_name,
                'total_trades': total_trades,
                'total_alerts': total_alerts,
                'alert_rate': alert_rate,
                'markets_tested': len(results),
                **{f'{det}_fires': count for det, count in detector_fires.items()}
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('alert_rate', ascending=False)
        
        return df
    
    def plot_alert_distribution(self, results: List, save_path: str = None):
        """Plot alert rate distribution across markets"""
        alert_rates = [
            r.alerts_generated / r.total_trades if r.total_trades > 0 else 0
            for r in results
        ]
        
        market_volumes = [r.total_trades for r in results]
        market_slugs = [r.market_slug[:30] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Alert rate by market
        ax1.barh(range(len(alert_rates)), alert_rates)
        ax1.set_yticks(range(len(alert_rates)))
        ax1.set_yticklabels(market_slugs, fontsize=8)
        ax1.set_xlabel('Alert Rate')
        ax1.set_title('Alert Rate by Market')
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Alert rate vs volume
        ax2.scatter(market_volumes, alert_rates, alpha=0.6, s=100)
        ax2.set_xlabel('Total Trades')
        ax2.set_ylabel('Alert Rate')
        ax2.set_title('Alert Rate vs Market Volume')
        ax2.set_xscale('log')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved plot to {save_path}")
        
        plt.show()
    
    def plot_score_distribution(self, results: List, save_path: str = None):
        """Plot total score distribution to check calibration"""
        all_scores = []
        
        for result in results:
            for features in result.all_trade_features:
                if features['total_score'] > 0:  # Only non-zero scores
                    all_scores.append(features['total_score'])
        
        if len(all_scores) == 0:
            logging.warning("No scores to plot!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Histogram
        ax1.hist(all_scores, bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(0.5, color='red', linestyle='--', label='Alert Threshold (0.5)')
        ax1.set_xlabel('Total Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Score Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Cumulative distribution
        sorted_scores = np.sort(all_scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        ax2.plot(sorted_scores, cumulative, linewidth=2)
        ax2.axvline(0.5, color='red', linestyle='--', label='Alert Threshold')
        ax2.set_xlabel('Total Score')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Score Distribution')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved plot to {save_path}")
        
        plt.show()
    
    def analyze_detector_coverage(self, results: List) -> pd.DataFrame:
        """
        Analyze which combinations of detectors fire together.
        Identifies redundant vs complementary detectors.
        """
        # Collect all alerts
        all_alerts = []
        for result in results:
            for alert in result.alerts:
                detector_names = [s.detector_name for s in alert.signals]
                all_alerts.append({
                    'num_detectors': len(detector_names),
                    'detectors': ', '.join(sorted(detector_names)),
                    'score': alert.total_score,
                    'market': result.market_slug
                })
        
        if len(all_alerts) == 0:
            logging.warning("No alerts to analyze!")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_alerts)
        
        # Count detector combinations
        combo_counts = df['detectors'].value_counts().head(20)
        
        print("\nTop 20 Detector Combinations:")
        for combo, count in combo_counts.items():
            print(f"  {count:4d}x | {combo}")
        
        # Single detector alerts vs multi-detector
        single_detector = (df['num_detectors'] == 1).sum()
        multi_detector = (df['num_detectors'] > 1).sum()
        
        print(f"\nAlert Composition:")
        print(f"  Single detector: {single_detector} ({single_detector/len(df)*100:.1f}%)")
        print(f"  Multi detector:  {multi_detector} ({multi_detector/len(df)*100:.1f}%)")
        
        return df
    
    def export_flagged_wallets(self, results: List, top_n: int = 50, 
                               output_path: str = None) -> pd.DataFrame:
        """
        Export top flagged wallets for manual labeling.
        """
        wallet_stats = {}
        
        for result in results:
            for alert in result.alerts:
                wallet = alert.trade.wallet
                
                if wallet not in wallet_stats:
                    wallet_stats[wallet] = {
                        'wallet': wallet,
                        'alert_count': 0,
                        'total_score_sum': 0,
                        'markets': set(),
                        'total_notional': 0,
                        'detectors_triggered': set(),
                    }
                
                wallet_stats[wallet]['alert_count'] += 1
                wallet_stats[wallet]['total_score_sum'] += alert.total_score
                wallet_stats[wallet]['markets'].add(result.market_slug)
                wallet_stats[wallet]['total_notional'] += alert.trade.notional_usdc
                
                for signal in alert.signals:
                    wallet_stats[wallet]['detectors_triggered'].add(signal.detector_name)
        
        # Convert to DataFrame
        rows = []
        for wallet, stats in wallet_stats.items():
            rows.append({
                'wallet': wallet,
                'alert_count': stats['alert_count'],
                'avg_score': stats['total_score_sum'] / stats['alert_count'],
                'markets_affected': len(stats['markets']),
                'total_notional': stats['total_notional'],
                'unique_detectors': len(stats['detectors_triggered']),
                'detectors': ', '.join(sorted(stats['detectors_triggered'])),
                'sample_markets': ', '.join(list(stats['markets'])[:3])
            })
        
        df = pd.DataFrame(rows)
        
        # Rank by composite score
        df['rank_score'] = (
            df['alert_count'] * 0.4 +
            df['avg_score'] * 30 * 0.3 +  # Scale to similar range
            df['markets_affected'] * 5 * 0.3
        )
        
        df = df.sort_values('rank_score', ascending=False).head(top_n)
        df['rank'] = range(1, len(df) + 1)
        
        # Add manual label column
        df['manual_label'] = '' 
        df['notes'] = ''
        
        # Reorder columns
        cols = ['rank', 'wallet', 'alert_count', 'avg_score', 'markets_affected',
                'total_notional', 'unique_detectors', 'detectors', 'sample_markets',
                'manual_label', 'notes']
        df = df[cols]
        
        if output_path:
            df.to_csv(output_path, index=False)
            logging.info(f"Exported {len(df)} wallets to {output_path}")
        
        return df
    
    def generate_summary_report(self, results: List, config_name: str = "default",
                               output_path: str = None):
        """
        Generate comprehensive summary report.
        """
        report = []
        report.append("="*80)
        report.append(f"BACKTEST SUMMARY REPORT: {config_name}")
        report.append("="*80)
        
        # Overall stats
        total_markets = len(results)
        total_trades = sum(r.total_trades for r in results)
        total_alerts = sum(r.alerts_generated for r in results)
        alert_rate = total_alerts / total_trades if total_trades > 0 else 0
        
        report.append(f"\nOverall Statistics:")
        report.append(f"  Markets tested: {total_markets}")
        report.append(f"  Total trades: {total_trades:,}")
        report.append(f"  Total alerts: {total_alerts:,}")
        report.append(f"  Alert rate: {alert_rate:.2%}")
        
        # Per-market breakdown
        report.append(f"\nPer-Market Results:")
        for result in sorted(results, key=lambda r: r.alerts_generated, reverse=True):
            rate = result.alerts_generated / result.total_trades if result.total_trades > 0 else 0
            report.append(f"  {result.market_slug[:60]:60s} | "
                         f"Trades: {result.total_trades:6,} | "
                         f"Alerts: {result.alerts_generated:5,} ({rate:5.2%})")
        
        # Detector statistics
        detector_totals = {}
        for result in results:
            for detector, count in result.detector_stats.items():
                detector_totals[detector] = detector_totals.get(detector, 0) + count
        
        report.append(f"\nDetector Fire Counts:")
        for detector, count in sorted(detector_totals.items(), key=lambda x: -x[1]):
            rate = count / total_trades if total_trades > 0 else 0
            report.append(f"  {detector:35s}: {count:6,} ({rate:5.2%})")
        
        report_text = '\n'.join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logging.info(f"Saved report to {output_path}")
        
        print(report_text)
        return report_text


# Example usage
if __name__ == "__main__":
    from backtesting.data_loader import HistoricalDataLoader
    from backtesting.backtest_runner import BacktestRunner
    from config import CONFIG
    
    logging.basicConfig(level=logging.INFO)
    
    # Load and run backtests
    loader = HistoricalDataLoader(data_dir="data")
    loader.load_data()
    
    markets = loader.get_markets_by_volume(min_volume=100000, limit=10)
    runner = BacktestRunner(CONFIG)
    
    results = []
    for market_id in markets:
        trades = loader.get_trades_for_market(market_id)
        metadata = loader.get_market_metadata(market_id)
        result = runner.run_backtest(trades, metadata)
        results.append(result)
    
    # Analyze
    analyzer = BacktestAnalyzer()
    
    # Generate summary
    analyzer.generate_summary_report(results, "Baseline", "summary_report.txt")
    
    # Plot distributions
    analyzer.plot_alert_distribution(results, "alert_distribution.png")
    analyzer.plot_score_distribution(results, "score_distribution.png")
    
    # Analyze detector coverage
    analyzer.analyze_detector_coverage(results)
    
    # Export flagged wallets for manual labeling
    flagged = analyzer.export_flagged_wallets(results, top_n=50, 
                                              output_path="flagged_wallets_for_labeling.csv")
    
    print("\nTop 10 Flagged Wallets:")
    print(flagged[['rank', 'wallet', 'alert_count', 'avg_score', 'markets_affected']].head(10))