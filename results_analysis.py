# -*- coding: utf-8 -*-
# @Date    : 1/13/2026
# @Author  : Jingwen
# @Desc    : Results visualization for WirelessAgent on WCHW dataset

"""
Visualization script for WirelessAgent performance analysis.
Generates publication-quality figures for:
1. Performance curve across optimization rounds
2. Monte Carlo Tree Search structure visualization
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Set publication-quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_results(results_path: str) -> dict:
    """Load and process results.json"""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get best score for each round (handling duplicates)
    round_scores = defaultdict(list)
    for entry in data:
        round_scores[entry['round']].append({
            'score': entry['score'],
            'cost': entry.get('total_cost', 0),
            'time': entry.get('time', '')
        })
    
    # Take the first (original) score for each round
    processed = {}
    for r in sorted(round_scores.keys()):
        processed[r] = round_scores[r][0]
    
    return processed


def load_experience(experience_path: str) -> dict:
    """Load and process experience.json"""
    with open(experience_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_performance_curve(results: dict, save_path: str):
    """
    Plot performance curve across optimization rounds.
    Shows score progression with confidence band and key milestones.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    rounds = sorted(results.keys())
    scores = [results[r]['score'] * 100 for r in rounds]  # Convert to percentage
    costs = [results[r]['cost'] for r in rounds]
    
    # Main performance line
    color_main = '#2E86AB'
    ax1.plot(rounds, scores, 'o-', color=color_main, linewidth=2, markersize=8, 
             label='Validation Accuracy', zorder=3)
    
    # Fill area under curve
    ax1.fill_between(rounds, scores, alpha=0.15, color=color_main)
    
    # Highlight best performance
    best_round = rounds[np.argmax(scores)]
    best_score = max(scores)
    ax1.scatter([best_round], [best_score], s=200, c='#E94F37', marker='*', 
                zorder=5, label=f'Best: {best_score:.2f}% (Round {best_round})')
    
    # Add baseline reference
    baseline_score = scores[0]
    ax1.axhline(y=baseline_score, color='gray', linestyle='--', alpha=0.7, 
                label=f'Baseline: {baseline_score:.2f}%')
    
    # Calculate improvement
    improvement = best_score - baseline_score
    ax1.annotate(f'+{improvement:.1f}%', 
                 xy=(best_round, best_score), 
                 xytext=(best_round + 1, best_score + 2),
                 fontsize=11, fontweight='bold', color='#E94F37',
                 arrowprops=dict(arrowstyle='->', color='#E94F37', lw=1.5))
    
    # Secondary axis for cost
    ax2 = ax1.twinx()
    color_cost = '#A23B72'
    ax2.bar(rounds, costs, alpha=0.3, color=color_cost, width=0.6, label='API Cost')
    ax2.set_ylabel('Total API Cost (USD)', color=color_cost, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_cost)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(color_cost)
    
    # Labels and formatting
    ax1.set_xlabel('Optimization Round', fontsize=12)
    ax1.set_ylabel('Validation Accuracy (%)', color=color_main, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_main)
    ax1.set_xlim(0.5, max(rounds) + 0.5)
    ax1.set_ylim(60, 85)
    ax1.set_xticks(rounds)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', framealpha=0.9)
    
    ax1.set_title('WirelessAgent Performance on WCHW Dataset', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def extract_modification_keywords(modification: str) -> str:
    """Extract key modification keywords for edge labels."""
    keywords = []
    mod_lower = modification.lower()
    
    # Key action patterns
    if 'toolagent' in mod_lower or 'tool agent' in mod_lower:
        keywords.append('+ToolAgent')
    if 'formula' in mod_lower:
        keywords.append('+Formula')
    if 'unit' in mod_lower:
        keywords.append('+Unit')
    if 'prompt' in mod_lower:
        keywords.append('Prompt')
    if 'ber' in mod_lower or 'q-function' in mod_lower or 'q function' in mod_lower:
        keywords.append('+BER/Q')
    if 'snr' in mod_lower or 'capacity' in mod_lower:
        keywords.append('+Capacity')
    if 'verification' in mod_lower or 'verify' in mod_lower:
        keywords.append('+Verify')
    if 'db' in mod_lower and 'convert' in mod_lower:
        keywords.append('+dB Conv')
    if 'water' in mod_lower and 'fill' in mod_lower:
        keywords.append('+WaterFill')
    
    if not keywords:
        # Fallback: extract first key phrase
        if len(modification) > 20:
            return modification[:18] + '...'
        return modification[:20] if modification else ''
    
    return ', '.join(keywords[:2])  # Max 2 keywords


def plot_mcts_tree(experience: dict, results: dict, save_path: str):
    """
    Plot Monte Carlo Tree Search structure showing parent-child relationships.
    Visualizes exploration and exploitation patterns with modification labels.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Build tree structure from experience with modification info
    nodes = {}  # round -> (x, y, score, parent)
    edges = []  # (parent_round, child_round, edge_type, score, modification)
    
    # Extract relationships
    for parent_str, exp_data in experience.items():
        parent = int(parent_str)
        parent_score = exp_data['score']
        
        # Success children
        for child_str, child_data in exp_data.get('success', {}).items():
            child = int(child_str)
            mod = child_data.get('modification', '')
            edges.append((parent, child, 'success', child_data['score'], mod))
        
        # Failure children
        for child_str, child_data in exp_data.get('failure', {}).items():
            child = int(child_str)
            mod = child_data.get('modification', '')
            edges.append((parent, child, 'failure', child_data['score'], mod))
        
        # Neutral children
        for child_str, child_data in exp_data.get('neutral', {}).items():
            child = int(child_str)
            mod = child_data.get('modification', '')
            edges.append((parent, child, 'neutral', child_data['score'], mod))
    
    # Position nodes in layers based on round number
    all_rounds = set([e[0] for e in edges] + [e[1] for e in edges])
    if results:
        all_rounds.update(results.keys())
    
    # Assign positions
    round_to_pos = {}
    layer_counts = defaultdict(int)
    
    # Sort rounds and assign y-positions (layers)
    sorted_rounds = sorted(all_rounds)
    for r in sorted_rounds:
        score = results[r]['score'] * 100 if r in results else 70
        # x based on score, y based on round
        x = score
        y = -r  # Negative so tree grows downward
        round_to_pos[r] = (x, y)
    
    # Draw edges first (behind nodes)
    edge_colors = {'success': '#27AE60', 'failure': '#C0392B', 'neutral': '#2980B9'}
    
    for parent, child, edge_type, child_score, modification in edges:
        if parent in round_to_pos and child in round_to_pos:
            px, py = round_to_pos[parent]
            cx, cy = round_to_pos[child]
            
            # Draw edge with arrow
            ax.annotate('', xy=(cx, cy), xytext=(px, py),
                       arrowprops=dict(arrowstyle='-|>', color=edge_colors[edge_type],
                                      lw=2.5, alpha=0.8,
                                      connectionstyle='arc3,rad=0.15',
                                      mutation_scale=15))
            
            # Only add modification label for success edges (performance improvement)
            if edge_type == 'success':
                mid_x = (px + cx) / 2
                mid_y = (py + cy) / 2
                
                # Extract keywords from modification
                label = extract_modification_keywords(modification)
                if label:
                    # Small offset to place label close to edge, Y offset down by 10%
                    offset_x = 0.2 if cx > px else -0.5
                    y_offset = -0.2 * abs(cy - py)  # Move down by 10% of edge length
                    ax.text(mid_x + offset_x, mid_y + y_offset, label, 
                           fontsize=10, ha='center', va='center',
                           color='black', fontweight='bold')
    
    # Draw nodes (larger size)
    for r, (x, y) in round_to_pos.items():
        score = results[r]['score'] * 100 if r in results else 70
        
        # Color based on score
        if score >= 80:
            color = '#27AE60'  # Green for high scores
        elif score >= 75:
            color = '#F39C12'  # Orange for medium
        elif score >= 70:
            color = '#3498DB'  # Blue for baseline
        else:
            color = '#E74C3C'  # Red for low
        
        # Larger size for better visibility
        is_parent = any(e[0] == r for e in edges)
        size = 1800 if is_parent else 1200
        
        ax.scatter(x, y, s=size, c=color, edgecolors='white', linewidths=3, zorder=3)
        # Round label inside circle (abbreviated)
        ax.annotate(f'Rd{r}', (x, y), ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white', zorder=4)
        # Accuracy label outside circle (below) - black text, no box
        ax.annotate(f'{score:.1f}%', (x, y - 0.6), ha='center', va='top',
                   fontsize=12, fontweight='bold', color='black', zorder=4)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#27AE60', label='Score ≥ 80%'),
        mpatches.Patch(color='#F39C12', label='75% ≤ Score < 80%'),
        mpatches.Patch(color='#3498DB', label='70% ≤ Score < 75%'),
        mpatches.Patch(color='#E74C3C', label='Score < 70%'),
        plt.Line2D([0], [0], color='#2ECC71', lw=2, label='Success (improved)'),
        plt.Line2D([0], [0], color='#E74C3C', lw=2, linestyle='--', label='Failure (degraded)'),
        plt.Line2D([0], [0], color='#3498DB', lw=2, linestyle=':', label='Neutral (similar)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=9)
    
    ax.set_xlabel('Validation Accuracy (%)', fontsize=12)
    ax.set_ylabel('Optimization Round', fontsize=12)
    ax.set_title('MCTS Exploration Tree for Workflow Optimization', fontsize=14, fontweight='bold', pad=15)
    
    # Format y-axis to show positive round numbers
    ax.set_yticks([-r for r in sorted_rounds])
    ax.set_yticklabels([str(r) for r in sorted_rounds])
    ax.set_xlim(58, 85)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_score_distribution(results: dict, save_path: str):
    """
    Plot score distribution and statistical analysis.
    Shows histogram, box plot, and key statistics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scores = [results[r]['score'] * 100 for r in sorted(results.keys())]
    rounds = sorted(results.keys())
    
    # Left: Box plot with individual points
    ax1 = axes[0]
    bp = ax1.boxplot(scores, patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#3498DB')
    bp['boxes'][0].set_alpha(0.6)
    bp['medians'][0].set_color('#E74C3C')
    bp['medians'][0].set_linewidth(2)
    
    # Scatter individual points
    jitter = np.random.normal(1, 0.04, len(scores))
    ax1.scatter(jitter, scores, alpha=0.7, c='#2E86AB', s=80, zorder=3, edgecolors='white')
    
    # Add statistics text
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    
    stats_text = f'Mean: {mean_score:.2f}%\nStd: {std_score:.2f}%\nMax: {max_score:.2f}%\nMin: {min_score:.2f}%'
    ax1.text(1.4, mean_score, stats_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.axhline(y=mean_score, color='#E74C3C', linestyle='--', alpha=0.7, label='Mean')
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax1.set_title('Score Distribution', fontsize=13, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_xlim(0.5, 1.8)
    
    # Right: Cumulative improvement
    ax2 = axes[1]
    baseline = scores[0]
    improvements = [s - baseline for s in scores]
    
    colors = ['#2ECC71' if imp > 0 else '#E74C3C' for imp in improvements]
    bars = ax2.bar(rounds, improvements, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Optimization Round', fontsize=12)
    ax2.set_ylabel('Improvement over Baseline (%)', fontsize=12)
    ax2.set_title('Performance Improvement by Round', fontsize=13, fontweight='bold')
    ax2.set_xticks(rounds)
    
    # Add best improvement annotation
    best_idx = np.argmax(improvements)
    best_round = rounds[best_idx]
    best_improvement = improvements[best_idx]
    ax2.annotate(f'+{best_improvement:.1f}%', 
                 xy=(best_round, best_improvement), 
                 xytext=(best_round, best_improvement + 2),
                 ha='center', fontsize=10, fontweight='bold', color='#27AE60')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ECC71', label='Improvement'),
        mpatches.Patch(color='#E74C3C', label='Degradation'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    plt.suptitle('WirelessAgent Performance Analysis on WCHW Dataset', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_method_comparison(save_path: str):
    """
    Plot bar chart comparing different methods on WCHW test set.
    Includes baselines and our WirelessAgent performance.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Method names and their test accuracy
    methods = [
        'Original\n(Qwen-Turbo)',
        'CoT\n(Wei et al., 2022)',
        'CoT-SC (5)\n(Wang et al., 2022)',
        'MedPrompt\n(Nori et al., 2023)',
        'ADAS\n(Hu et al., 2024)',
        'AFlow\n(Zhang et al., 2025)',
        'WirelessAgent\n(Ours)'
    ]
    
    scores = [0.58343, 0.6032, 0.6001, 0.6122, 0.5313, 0.6992, 0.7794]
    scores_pct = [s * 100 for s in scores]  # Convert to percentage
    
    # Colors: gray for baselines, highlight for ours
    colors = ['#95A5A6', '#95A5A6', '#95A5A6', '#95A5A6', '#95A5A6', '#3498DB', '#27AE60']
    
    # Create bars
    x = np.arange(len(methods))
    bars = ax.bar(x, scores_pct, color=colors, edgecolor='white', linewidth=2, width=0.7)
    
    # Add value labels on top of bars
    for i, (bar, score) in enumerate(zip(bars, scores_pct)):
        height = bar.get_height()
        fontweight = 'bold' if i == len(methods) - 1 else 'normal'
        color = '#27AE60' if i == len(methods) - 1 else 'black'
        ax.annotate(f'{score:.2f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=11, fontweight=fontweight, color=color)
    
    # Highlight our method with a star
    best_idx = len(methods) - 1
    ax.scatter([best_idx], [scores_pct[best_idx] + 3], marker='*', s=200, c='#E74C3C', zorder=5)
    
    # Calculate improvement over best baseline
    best_baseline = max(scores_pct[:-1])
    our_score = scores_pct[-1]
    improvement = our_score - best_baseline
    
    # Add improvement annotation
    ax.annotate(f'+{improvement:.2f}%',
               xy=(best_idx, our_score + 5),
               xytext=(best_idx - 0.5, our_score + 10),
               fontsize=12, fontweight='bold', color='#E74C3C',
               arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))
    
    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylim(0, 90)
    ax.set_title('Method Comparison on WCHW Test Set', fontsize=14, fontweight='bold', pad=15)
    
    # Add horizontal line for our method
    ax.axhline(y=our_score, color='#27AE60', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    """Main function to generate all visualizations."""
    # Paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_path, 'workspace', 'WCHW', 'workflows', 'results.json')
    experience_path = os.path.join(base_path, 'workspace', 'WCHW', 'workflows', 'processed_experience.json')
    assets_path = os.path.join(base_path, 'assets')
    
    # Ensure assets directory exists
    os.makedirs(assets_path, exist_ok=True)
    
    # Load data
    print("Loading data...")
    results = load_results(results_path)
    experience = load_experience(experience_path)
    
    print(f"Loaded {len(results)} rounds of results")
    print(f"Loaded experience data for {len(experience)} parent rounds")
    
    # Generate figures
    print("\nGenerating visualizations...")
    
    # Figure 1: Performance curve
    plot_performance_curve(
        results, 
        os.path.join(assets_path, 'performance_curve.png')
    )
    
    # Figure 2: MCTS tree structure
    plot_mcts_tree(
        experience, 
        results,
        os.path.join(assets_path, 'mcts_tree.png')
    )
    
    # Figure 3: Score distribution and improvement analysis
    plot_score_distribution(
        results,
        os.path.join(assets_path, 'score_analysis.png')
    )
    
    # Figure 4: Method comparison bar chart
    plot_method_comparison(
        os.path.join(assets_path, 'method_comparison.png')
    )
    
    print("\n✅ All visualizations saved to 'assets/' directory")
    print("   - performance_curve.png: Performance across optimization rounds")
    print("   - mcts_tree.png: MCTS exploration tree structure")
    print("   - score_analysis.png: Score distribution and improvement analysis")
    print("   - method_comparison.png: Comparison with baseline methods")


if __name__ == "__main__":
    main()
