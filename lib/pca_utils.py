# PCA Feature Engineering Utilities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_pca_config(pca_cfg, config_path):
    """Load PCA feature groups from CSV"""
    pca_features_file = f"{config_path}/{pca_cfg['features_file']}"
    pca_df = pd.read_csv(pca_features_file, comment='#', encoding='latin-1')
    group_col = pca_cfg['group_column']
    pca_groups = pca_df[pca_df[group_col].notna() & (pca_df[group_col] != '')][group_col].unique()
    return pca_df, sorted(pca_groups), group_col


def apply_pca_to_group(data, columns, variance_threshold=0.95, max_components=None):
    """Apply PCA to a set of columns"""
    cols_available = [c for c in columns if c in data.columns]
    if len(cols_available) < 2:
        raise ValueError(f"Need 2+ columns, found {len(cols_available)}")
    X = data[cols_available].copy().fillna(0)
    variances = X.var()
    non_constant_cols = variances[variances > 0].index.tolist()
    if len(non_constant_cols) < 2:
        raise ValueError(f"Only {len(non_constant_cols)} non-constant columns")
    X = X[non_constant_cols]
    
    # Standardize features (mean=0, std=1) before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine n_components: use max_components if set, otherwise variance_threshold
    if max_components is not None:
        n_components = min(max_components, len(non_constant_cols))
    else:
        n_components = variance_threshold
    
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return pca, X_pca, non_constant_cols


def apply_pca_groups(data, pca_cfg, config_path):
    """Apply PCA to all configured groups"""
    pca_df, pca_groups, group_col = load_pca_config(pca_cfg, config_path)
    variance_threshold = pca_cfg.get('variance_threshold', 0.95)
    max_components = pca_cfg.get('max_components', None)
    prefix_template = pca_cfg.get('prefix_template', 'pca_{group}_')
    pca_results = {}
    all_pca_features = pd.DataFrame()
    
    # Print configuration
    print(f"\nApplying PCA to {len(pca_groups)} groups...")
    if max_components is not None:
        print(f"  Max components per group: {max_components}")
    else:
        print(f"  Variance threshold: {variance_threshold}")
    
    for group in pca_groups:
        print(f"\n{'='*60}\nPCA Group: {group}\n{'='*60}")
        cols = pca_df[pca_df[group_col] == group]['column_name'].tolist()
        try:
            pca, X_pca, feature_columns = apply_pca_to_group(
                data, cols, variance_threshold, max_components
            )
            n_components = pca.n_components_
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            print(f"  Input columns: {len(cols)}")
            print(f"  Components kept: {n_components}")
            print(f"  Cumulative variance: {cumulative_var[-1]:.1%}")
            prefix = prefix_template.format(group=group)
            pca_col_names = [f'{prefix}{i}' for i in range(n_components)]
            pca_features = pd.DataFrame(X_pca, columns=pca_col_names, index=data.index)
            if all_pca_features.empty:
                all_pca_features = pca_features
            else:
                all_pca_features = pd.concat([all_pca_features, pca_features], axis=1)
            print(f"  New features: {', '.join(pca_col_names)}")
            pca_results[group] = {
                'n_components': int(n_components),
                'explained_variance_ratio': explained_var.tolist(),
                'cumulative_variance': float(cumulative_var[-1]),
                'feature_names': pca_col_names,
                'input_columns': feature_columns,
                'pca_object': pca
            }
        except ValueError as e:
            print(f"  Skip: {e}")
    print(f"\n{'='*60}\nPCA complete: {all_pca_features.shape[1]} features")
    return pca_results, all_pca_features


def create_scree_plot(pca_result, plots_dir, group_name):
    """Create and save scree plot"""
    pca = pca_result['pca_object']
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    n_comp = len(explained_var)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, n_comp + 1)
    ax.bar(x, explained_var, alpha=0.7, color='steelblue', label='Individual')
    ax2 = ax.twinx()
    ax2.plot(x, cumulative_var, 'ro-', linewidth=2, markersize=6, label='Cumulative')
    ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95%')
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax2.set_ylabel('Cumulative Variance Ratio', fontsize=12)
    ax.set_title(f'PCA Scree Plot: {group_name}', fontsize=14, fontweight='bold')
    if n_comp <= 20:
        ax.set_xticks(x)
    ax.set_ylim([0, max(explained_var) * 1.1])
    ax2.set_ylim([0, 1.05])
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    scree_path = f'{plots_dir}/pca_{group_name}_scree.png'
    fig.savefig(scree_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return scree_path


def create_loadings_heatmap(pca_result, plots_dir, group_name):
    """Create and save loadings heatmap"""
    pca = pca_result['pca_object']
    feature_cols = pca_result['input_columns']
    n_comp = pca_result['n_components']
    loadings = pca.components_.T
    loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(n_comp)], index=feature_cols)
    if len(feature_cols) > 30:
        top_features = set()
        for i in range(min(3, n_comp)):
            abs_loadings = loadings_df[f'PC{i+1}'].abs()
            top_features.update(abs_loadings.nlargest(10).index)
        loadings_df = loadings_df.loc[list(top_features)]
    fig, ax = plt.subplots(figsize=(max(8, n_comp * 1.5), max(8, len(loadings_df) * 0.3)))
    sns.heatmap(loadings_df, cmap='RdBu_r', center=0, cbar_kws={'label': 'Loading'}, ax=ax)
    ax.set_title(f'PCA Loadings: {group_name}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    loadings_path = f'{plots_dir}/pca_{group_name}_loadings.png'
    fig.savefig(loadings_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return loadings_path


def save_pca_summary(pca_results, results_dir):
    """Save PCA summary to YAML"""
    import yaml
    summary = {'enabled': True, 'groups': {}}
    for group, result in pca_results.items():
        summary['groups'][group] = {
            'n_components': result['n_components'],
            'cumulative_variance': result['cumulative_variance'],
            'feature_names': result['feature_names']
        }
    summary_file = f"{results_dir}/04a_pca_summary.yaml"
    with open(summary_file, 'w') as f:
        yaml.dump(summary, f)
    return summary_file
