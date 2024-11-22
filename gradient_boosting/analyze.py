import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    def __init__(self, target_column='User_Type'):
        self.target_column = target_column
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare_data(self, data_path):
        """Load and prepare initial dataset"""
        print("Loading data...")
        data = pd.read_csv(data_path)
        
        # Handle missing values
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
        # Create basic derived features
        data['Charging_Start_Time'] = pd.to_datetime(data['Charging Start Time'])
        data['Hour'] = data['Charging_Start_Time'].dt.hour
        data['DayOfWeek'] = data['Charging_Start_Time'].dt.dayofweek
        data['Weekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
        
        data['Energy_Per_km'] = data['Energy Consumed (kWh)'] / np.maximum(data['Distance Driven (since last charge) (km)'], 1)
        data['Cost_Per_km'] = data['Charging Cost (USD)'] / np.maximum(data['Distance Driven (since last charge) (km)'], 1)
        
        return data
    
    def analyze_feature_distributions(self, data, feature):
        """Analyze distribution of a feature across user types"""
        plt.figure(figsize=(15, 5))
        
        # Distribution plot
        plt.subplot(1, 3, 1)
        for user_type in data[self.target_column].unique():
            sns.kdeplot(data=data[data[self.target_column] == user_type][feature], 
                       label=user_type)
        plt.title(f'Distribution of {feature} by User Type')
        plt.legend()
        
        # Box plot
        plt.subplot(1, 3, 2)
        sns.boxplot(data=data, x=self.target_column, y=feature)
        plt.xticks(rotation=45)
        plt.title(f'Box Plot of {feature}')
        
        # Violin plot
        plt.subplot(1, 3, 3)
        sns.violinplot(data=data, x=self.target_column, y=feature)
        plt.xticks(rotation=45)
        plt.title(f'Violin Plot of {feature}')
        
        plt.tight_layout()
        plt.show()
        
        # Statistical tests
        print(f"\nStatistical Analysis for {feature}:")
        print("\nSummary Statistics by User Type:")
        print(data.groupby(self.target_column)[feature].describe())
        
        # One-way ANOVA
        user_types = data[self.target_column].unique()
        groups = [data[data[self.target_column] == ut][feature].values for ut in user_types]
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"\nOne-way ANOVA:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        # Effect size (Eta-squared)
        def eta_squared(f_stat, groups):
            n = sum(len(group) for group in groups)
            k = len(groups)
            return (f_stat * (k-1)) / (f_stat * (k-1) + (n-k))
        
        eta_sq = eta_squared(f_stat, groups)
        print(f"Effect size (η²): {eta_sq:.4f}")
        
    def analyze_temporal_patterns(self, data):
        """Analyze temporal patterns in charging behavior"""
        plt.figure(figsize=(15, 10))
        
        # Hourly patterns
        plt.subplot(2, 2, 1)
        hourly_counts = pd.crosstab(data['Hour'], data[self.target_column], normalize='columns')
        hourly_counts.plot(kind='line', marker='o')
        plt.title('Hourly Charging Patterns by User Type')
        plt.xlabel('Hour of Day')
        plt.ylabel('Proportion of Charges')
        
        # Daily patterns
        plt.subplot(2, 2, 2)
        daily_counts = pd.crosstab(data['DayOfWeek'], data[self.target_column], normalize='columns')
        daily_counts.plot(kind='line', marker='o')
        plt.title('Daily Charging Patterns by User Type')
        plt.xlabel('Day of Week')
        plt.ylabel('Proportion of Charges')
        
        # Energy consumption by hour
        plt.subplot(2, 2, 3)
        sns.boxplot(data=data, x='Hour', y='Energy Consumed (kWh)', hue=self.target_column)
        plt.title('Energy Consumption by Hour and User Type')
        plt.xticks(rotation=45)
        
        # Distance by day
        plt.subplot(2, 2, 4)
        sns.boxplot(data=data, x='DayOfWeek', y='Distance Driven (since last charge) (km)', 
                   hue=self.target_column)
        plt.title('Distance Driven by Day and User Type')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def analyze_feature_correlations(self, data):
        """Analyze correlations between features and identify patterns"""
        # Select numerical columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'User ID']
        
        # Calculate correlations
        corr_matrix = data[numeric_cols].corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.show()
        
        # Find strongly correlated features
        strong_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr = abs(corr_matrix.iloc[i,j])
                if corr > 0.7:
                    strong_correlations.append({
                        'Feature1': numeric_cols[i],
                        'Feature2': numeric_cols[j],
                        'Correlation': corr
                    })
        
        if strong_correlations:
            print("\nStrongly Correlated Features (|correlation| > 0.7):")
            print(pd.DataFrame(strong_correlations).sort_values('Correlation', ascending=False))
    
    def analyze_feature_importance(self, data):
        """Analyze feature importance using multiple methods"""
        # Prepare data
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['User ID', self.target_column]]
        
        X = data[numeric_cols]
        y = self.label_encoder.fit_transform(data[self.target_column])
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Mutual Information
        mi_scores = mutual_info_classif(X, y)
        
        # Combine and compare importance scores
        importance_df = pd.DataFrame({
            'Feature': numeric_cols,
            'RF_Importance': rf.feature_importances_,
            'MI_Score': mi_scores
        })
        
        # Sort by average importance
        importance_df['Avg_Importance'] = (
            (importance_df['RF_Importance'] / importance_df['RF_Importance'].max()) +
            (importance_df['MI_Score'] / importance_df['MI_Score'].max())
        ) / 2
        
        importance_df = importance_df.sort_values('Avg_Importance', ascending=False)
        
        # Plot importance scores
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.barplot(data=importance_df.head(10), x='RF_Importance', y='Feature')
        plt.title('Top 10 Features by Random Forest Importance')
        
        plt.subplot(1, 2, 2)
        sns.barplot(data=importance_df.head(10), x='MI_Score', y='Feature')
        plt.title('Top 10 Features by Mutual Information')
        
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def analyze_class_separation(self, data):
        """Analyze how well different features separate the classes"""
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['User ID', self.target_column]]
        
        separation_scores = []
        
        for feature in numeric_cols:
            # Calculate class means and variances
            class_stats = data.groupby(self.target_column)[feature].agg(['mean', 'std'])
            
            # Calculate separation score (ratio of between-class to within-class variance)
            global_mean = data[feature].mean()
            between_class_var = sum((class_stats['mean'] - global_mean) ** 2) / len(class_stats)
            within_class_var = class_stats['std'].mean() ** 2
            
            separation_score = between_class_var / within_class_var if within_class_var != 0 else 0
            
            separation_scores.append({
                'Feature': feature,
                'Separation_Score': separation_score
            })
        
        separation_df = pd.DataFrame(separation_scores)
        separation_df = separation_df.sort_values('Separation_Score', ascending=False)
        
        # Plot top separating features
        plt.figure(figsize=(10, 6))
        sns.barplot(data=separation_df.head(15), x='Separation_Score', y='Feature')
        plt.title('Top 15 Features by Class Separation Score')
        plt.tight_layout()
        plt.show()
        
        return separation_df
    
    def run_complete_analysis(self, data_path):
        """Run complete feature analysis pipeline"""
        data = self.load_and_prepare_data(data_path)
        
        print("\nAnalyzing Feature Importance...")
        importance_df = self.analyze_feature_importance(data)
        
        print("\nAnalyzing Class Separation...")
        separation_df = self.analyze_class_separation(data)
        
        print("\nAnalyzing Temporal Patterns...")
        self.analyze_temporal_patterns(data)
        
        print("\nAnalyzing Feature Correlations...")
        self.analyze_feature_correlations(data)
        
        # Analyze distributions of top features
        top_features = importance_df['Feature'].head(5).tolist()
        print("\nAnalyzing Top Feature Distributions...")
        for feature in top_features:
            self.analyze_feature_distributions(data, feature)
        
        return importance_df, separation_df, data
    

class EnhancedPrintAnalyzer(FeatureAnalyzer):
    def print_basic_stats(self, data):
        """Print basic statistics about the dataset"""
        print("\n=== Basic Dataset Statistics ===")
        print(f"Total number of charging sessions: {len(data)}")
        print("\nSessions by User Type:")
        user_type_counts = data[self.target_column].value_counts()
        for user_type, count in user_type_counts.items():
            print(f"{user_type}: {count} sessions ({count/len(data)*100:.1f}%)")
        
        print("\nNumerical Features Summary:")
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['User ID']]
        print(data[numeric_cols].describe().round(2).to_string())
    
    def print_efficiency_metrics(self, data):
        """Print detailed efficiency metrics by user type"""
        print("\n=== Efficiency Metrics by User Type ===")
        metrics = {
            'Energy_Per_km': 'kWh/km',
            'Cost_Per_km': 'USD/km',
            'Energy Consumed (kWh)': 'kWh',
            'Charging Cost (USD)': 'USD'
        }
        
        for metric, unit in metrics.items():
            print(f"\n{metric} ({unit}):")
            stats = data.groupby(self.target_column)[metric].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(3)
            print(stats.to_string())
            
            # Calculate and print percentile statistics
            percentiles = data.groupby(self.target_column)[metric].describe(
                percentiles=[.25, .5, .75, .9]
            ).round(3)
            print("\nPercentiles:")
            print(percentiles[['25%', '50%', '75%', '90%']].to_string())
    
    def print_temporal_patterns(self, data):
        """Print temporal charging patterns"""
        print("\n=== Temporal Charging Patterns ===")
        
        # Peak charging hours
        print("\nPeak Charging Hours by User Type:")
        for user_type in data[self.target_column].unique():
            user_data = data[data[self.target_column] == user_type]
            peak_hour = user_data['Hour'].mode().iloc[0]
            count = len(user_data[user_data['Hour'] == peak_hour])
            percentage = (count / len(user_data)) * 100
            print(f"{user_type}: Hour {peak_hour:02d}:00 ({percentage:.1f}% of sessions)")
        
        # Weekend vs Weekday patterns
        print("\nWeekend vs Weekday Charging:")
        weekend_stats = data.groupby([self.target_column, 'Weekend']).agg({
            'Energy Consumed (kWh)': 'mean',
            'Charging Cost (USD)': 'mean',
            'Distance Driven (since last charge) (km)': 'mean'
        }).round(2)
        
        print("\nAverage metrics for weekday/weekend:")
        print(weekend_stats.to_string())
    
    def print_behavioral_insights(self, data):
        """Print behavioral insights about different user types"""
        print("\n=== Behavioral Insights ===")
        
        # Distance patterns
        print("\nDistance Patterns:")
        distance_stats = data.groupby(self.target_column)['Distance Driven (since last charge) (km)'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        print("\nDistance Driven Between Charges (km):")
        print(distance_stats.to_string())
        
        # State of charge patterns
        print("\nCharging Behavior:")
        soc_stats = data.groupby(self.target_column).agg({
            'State of Charge (Start %)': ['mean', 'median'],
            'State of Charge (End %)': ['mean', 'median']
        }).round(2)
        print("\nTypical State of Charge (%):")
        print(soc_stats.to_string())
        
        # Charging duration patterns
        print("\nCharging Duration Patterns:")
        duration_stats = data.groupby(self.target_column)['Charging Duration (hours)'].agg([
            'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        print("\nCharging Duration (hours):")
        print(duration_stats.to_string())
        
        # Battery capacity utilization
        print("\nBattery Capacity Utilization:")
        capacity_stats = data.groupby(self.target_column).agg({
            'Battery Capacity (kWh)': ['mean', 'median'],
            'Energy Consumed (kWh)': ['mean', 'median']
        }).round(2)
        
        # Calculate utilization percentage
        capacity_stats['Utilization %'] = (
            capacity_stats['Energy Consumed (kWh)']['mean'] / 
            capacity_stats['Battery Capacity (kWh)']['mean'] * 100
        ).round(2)
        
        print("\nBattery Usage Patterns:")
        print(capacity_stats.to_string())
    
    def print_correlations(self, data):
        """Print important feature correlations"""
        print("\n=== Feature Correlations ===")
        
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['User ID']]
        
        corr_matrix = data[numeric_cols].corr()
        
        # Print strong correlations
        print("\nStrong Correlations (|correlation| > 0.5):")
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr = corr_matrix.iloc[i,j]
                if abs(corr) > 0.5:
                    print(f"{numeric_cols[i]} vs {numeric_cols[j]}: {corr:.3f}")
    
    def run_complete_analysis(self, data_path):
        """Run complete analysis with detailed printing"""
        data = self.load_and_prepare_data(data_path)
        
        # Print all analyses
        self.print_basic_stats(data)
        self.print_efficiency_metrics(data)
        self.print_temporal_patterns(data)
        self.print_behavioral_insights(data)
        self.print_correlations(data)
        
        return data
    
class AdvancedAnalyzer(EnhancedPrintAnalyzer):
    def print_statistical_tests(self, data):
        """Perform and print statistical tests to compare user types"""
        print("\n=== Statistical Analysis ===")
        
        # Key metrics to test
        metrics = {
            'Energy_Per_km': 'Energy Efficiency',
            'Cost_Per_km': 'Cost Efficiency',
            'Distance Driven (since last charge) (km)': 'Distance per Charge',
            'Charging Duration (hours)': 'Charging Duration',
            'State of Charge (Start %)': 'Starting State of Charge',
            'State of Charge (End %)': 'Ending State of Charge'
        }
        
        for col, name in metrics.items():
            print(f"\nAnalyzing {name}:")
            
            # Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA)
            groups = [group[col].values for name, group in data.groupby(self.target_column)]
            h_stat, p_value = stats.kruskal(*groups)
            
            print(f"Kruskal-Wallis H-test:")
            print(f"H-statistic: {h_stat:.4f}")
            print(f"p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                # If significant, perform pairwise Mann-Whitney U tests
                user_types = data[self.target_column].unique()
                print("\nPairwise Mann-Whitney U tests:")
                for i in range(len(user_types)):
                    for j in range(i+1, len(user_types)):
                        stat, p = stats.mannwhitneyu(
                            data[data[self.target_column] == user_types[i]][col],
                            data[data[self.target_column] == user_types[j]][col],
                            alternative='two-sided'
                        )
                        print(f"{user_types[i]} vs {user_types[j]}: p={p:.4f}")
    
    def analyze_efficiency_differences(self, data):
        """Analyze and print efficiency differences between user types"""
        print("\n=== Efficiency Analysis ===")
        
        # Calculate median values for key metrics
        metrics = ['Energy_Per_km', 'Cost_Per_km', 'Charging Rate (kW)']
        
        for metric in metrics:
            print(f"\n{metric} Analysis:")
            medians = data.groupby(self.target_column)[metric].median()
            print("\nMedian Values:")
            print(medians.to_string())
            
            # Calculate relative differences
            overall_median = data[metric].median()
            print("\nRelative to Overall Median (%):")
            relative_diff = ((medians - overall_median) / overall_median * 100).round(2)
            print(relative_diff.to_string())
    
    def analyze_usage_patterns(self, data):
        """Analyze and print distinct usage patterns"""
        print("\n=== Usage Pattern Analysis ===")
        
        # Define time periods
        data['TimeCategory'] = pd.cut(data['Hour'], 
                                    bins=[-1, 6, 12, 18, 24],
                                    labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Analyze charging patterns by time period
        time_patterns = pd.crosstab(data[self.target_column], 
                                  data['TimeCategory'], 
                                  normalize='index') * 100
        
        print("\nCharging Time Distribution (%):")
        print(time_patterns.round(2))
        
        # Analyze weekend vs weekday patterns in more detail
        weekend_patterns = data.groupby([self.target_column, 'Weekend']).agg({
            'Energy Consumed (kWh)': ['mean', 'std'],
            'Charging Duration (hours)': ['mean', 'std'],
            'Distance Driven (since last charge) (km)': ['mean', 'std']
        }).round(2)
        
        print("\nDetailed Weekend vs Weekday Patterns:")
        print(weekend_patterns.to_string())
    
    def run_complete_analysis(self, data_path):
        """Run complete analysis with additional statistical tests"""
        data = self.load_and_prepare_data(data_path)
        
        # Run original analyses
        self.print_basic_stats(data)
        self.print_efficiency_metrics(data)
        self.print_temporal_patterns(data)
        self.print_behavioral_insights(data)
        self.print_correlations(data)
        
        # Run additional analyses
        self.print_statistical_tests(data)
        self.analyze_efficiency_differences(data)
        self.analyze_usage_patterns(data)
        
        return data

class DetailedAnalyzer(AdvancedAnalyzer):
    def analyze_temperature_effects(self, data):
        """Analyze how temperature affects charging patterns"""
        print("\n=== Temperature Effects Analysis ===")
        
        # Create temperature bins
        data['Temp_Category'] = pd.cut(data['Temperature (°C)'],
                                     bins=[-20, 0, 15, 25, 100],
                                     labels=['Cold (< 0°C)', 'Cool (0-15°C)', 
                                           'Moderate (15-25°C)', 'Warm (>25°C)'])
        
        # Analyze key metrics by temperature
        metrics = {
            'Energy_Per_km': 'Energy Efficiency',
            'Charging Rate (kW)': 'Charging Rate',
            'Charging Duration (hours)': 'Charging Duration',
            'Energy Consumed (kWh)': 'Energy Consumed'
        }
        
        for metric, name in metrics.items():
            print(f"\n{name} by Temperature Category:")
            stats_df = data.groupby(['User Type', 'Temp_Category'])[metric].agg([
                'count', 'mean', 'std', 'median'
            ]).round(3)
            print(stats_df.to_string())
            
            # Perform Kruskal-Wallis test for temperature effect within each user type
            print(f"\nKruskal-Wallis test for temperature effect on {name}:")
            for user_type in data['User Type'].unique():
                user_data = data[data['User Type'] == user_type]
                groups = [group[metric].values for _, group in user_data.groupby('Temp_Category')]
                if len(groups) > 1:  # Check if we have data in multiple temperature categories
                    try:
                        h_stat, p_value = stats.kruskal(*[g for g in groups if len(g) > 0])
                        print(f"{user_type}: H-statistic = {h_stat:.4f}, p-value = {p_value:.4f}")
                    except ValueError as e:
                        print(f"{user_type}: Insufficient data for statistical test")

    def analyze_vehicle_age_efficiency(self, data):
        """Analyze relationship between vehicle age and efficiency"""
        print("\n=== Vehicle Age and Efficiency Analysis ===")
        
        # Create age categories
        data['Age_Category'] = pd.cut(data['Vehicle Age (years)'],
                                    bins=[-1, 2, 4, 6, 100],
                                    labels=['New (0-2)', 'Mid (2-4)', 
                                          'Mature (4-6)', 'Older (>6)'])
        
        # Analyze efficiency metrics by age
        metrics = ['Energy_Per_km', 'Cost_Per_km', 'Charging Rate (kW)']
        
        for metric in metrics:
            print(f"\n{metric} by Vehicle Age Category:")
            stats_df = data.groupby(['User Type', 'Age_Category'])[metric].agg([
                'count', 'mean', 'std', 'median'
            ]).round(3)
            print(stats_df.to_string())
        
        # Calculate correlation coefficients
        print("\nCorrelation with Vehicle Age:")
        for user_type in data['User Type'].unique():
            user_data = data[data['User Type'] == user_type]
            correlations = {
                metric: user_data['Vehicle Age (years)'].corr(user_data[metric])
                for metric in metrics
            }
            print(f"\n{user_type}:")
            for metric, corr in correlations.items():
                print(f"{metric}: {corr:.3f}")

    def analyze_charging_rate_patterns(self, data):
        """Detailed analysis of charging rate patterns"""
        print("\n=== Charging Rate Pattern Analysis ===")
        
        # Create charging rate categories
        data['Rate_Category'] = pd.cut(data['Charging Rate (kW)'],
                                     bins=[0, 15, 30, 50, 100],
                                     labels=['Low (<15kW)', 'Medium (15-30kW)', 
                                           'High (30-50kW)', 'Very High (>50kW)'])
        
        # Analyze charging rate distribution
        print("\nCharging Rate Distribution (%):")
        rate_dist = pd.crosstab(data['User Type'], 
                               data['Rate_Category'], 
                               normalize='index') * 100
        print(rate_dist.round(2))
        
        # Analyze efficiency by charging rate
        print("\nEfficiency by Charging Rate Category:")
        efficiency_stats = data.groupby(['User Type', 'Rate_Category'])['Energy_Per_km'].agg([
            'count', 'mean', 'std', 'median'
        ]).round(3)
        print(efficiency_stats.to_string())
        
        # Analyze time of day preferences for different charging rates
        print("\nCharging Rate Usage by Time of Day (%):")
        time_rate_dist = pd.crosstab([data['User Type'], data['TimeCategory']], 
                                    data['Rate_Category'], 
                                    normalize='index') * 100
        print(time_rate_dist.round(2))
        
        # Analyze correlation between charging rate and duration
        print("\nCorrelation between Charging Rate and Duration:")
        for user_type in data['User Type'].unique():
            user_data = data[data['User Type'] == user_type]
            corr = user_data['Charging Rate (kW)'].corr(user_data['Charging Duration (hours)'])
            print(f"{user_type}: {corr:.3f}")

    def run_complete_analysis(self, data_path):
        """Run all analyses including the new detailed ones"""
        data = self.load_and_prepare_data(data_path)
        
        # Run original analyses
        super().run_complete_analysis(data_path)
        
        # Run new detailed analyses
        self.analyze_temperature_effects(data)
        self.analyze_vehicle_age_efficiency(data)
        self.analyze_charging_rate_patterns(data)
        
        return data

if __name__ == "__main__":
    analyzer = DetailedAnalyzer(target_column='User Type')
    data = analyzer.run_complete_analysis('../ev_charging_patterns.csv')