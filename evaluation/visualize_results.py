import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def analyze_results(results_df: pd.DataFrame, category_name: str, output_dir: str):

    print(f"\n{'='*70}")
    print(f" {category_name.upper()} 카테고리 분석")
    print(f"{'='*70}")

    print(f"\n[기본 통계]")
    print(f"  평균 Recall: {results_df['recall'].mean():.4f}")
    print(f"  평균 Precision: {results_df['precision'].mean():.4f}")
    print(f"  평균 F1: {results_df['f1'].mean():.4f}")
    print(f"  최고 Recall: {results_df['recall'].max():.4f}")
    print(f"  최고 Precision: {results_df['precision'].max():.4f}")
    print(f"  최고 F1: {results_df['f1'].max():.4f}")

    print(f"\n[최적 설정]")

    best_recall = results_df.loc[results_df['recall'].idxmax()]
    print(f"\n  Best Recall: {best_recall['recall']:.4f}")
    print(f"    - chunk_size: {best_recall['chunk_size']}, overlap: {best_recall['chunk_overlap']}, top_k: {best_recall['top_k']}")

    best_precision = results_df.loc[results_df['precision'].idxmax()]
    print(f"\n  Best Precision: {best_precision['precision']:.4f}")
    print(f"    - chunk_size: {best_precision['chunk_size']}, overlap: {best_precision['chunk_overlap']}, top_k: {best_precision['top_k']}")

    best_f1 = results_df.loc[results_df['f1'].idxmax()]
    print(f"\n  Best F1: {best_f1['f1']:.4f}")
    print(f"    - chunk_size: {best_f1['chunk_size']}, overlap: {best_f1['chunk_overlap']}, top_k: {best_f1['top_k']}")

    print(f"\n[Chunk Size별 평균 성능]")
    chunk_stats = results_df.groupby('chunk_size').agg({
        'recall': 'mean',
        'precision': 'mean',
        'f1': 'mean'
    }).round(4)
    print(chunk_stats)

    print(f"\n[Top-k별 평균 성능]")
    topk_stats = results_df.groupby('top_k').agg({
        'recall': 'mean',
        'precision': 'mean',
        'f1': 'mean'
    }).round(4)
    print(topk_stats)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{category_name.upper()} 카테고리 평가 결과', fontsize=16)

    ax = axes[0, 0]
    chunk_recall = results_df.groupby('chunk_size')['recall'].mean()
    chunk_recall.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Chunk Size별 평균 Recall')
    ax.set_xlabel('Chunk Size')
    ax.set_ylabel('Recall')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    topk_recall = results_df.groupby('top_k')['recall'].mean()
    topk_recall.plot(kind='line', marker='o', ax=ax, color='steelblue')
    ax.set_title('Top-k별 평균 Recall')
    ax.set_xlabel('Top-k')
    ax.set_ylabel('Recall')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    pivot_recall = results_df.pivot_table(values='recall', index='chunk_size', columns='top_k', aggfunc='mean')
    sns.heatmap(pivot_recall, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax)
    ax.set_title('Recall Heatmap (Chunk Size vs Top-k)')

    ax = axes[1, 0]
    chunk_precision = results_df.groupby('chunk_size')['precision'].mean()
    chunk_precision.plot(kind='bar', ax=ax, color='coral')
    ax.set_title('Chunk Size별 평균 Precision')
    ax.set_xlabel('Chunk Size')
    ax.set_ylabel('Precision')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    topk_precision = results_df.groupby('top_k')['precision'].mean()
    topk_precision.plot(kind='line', marker='o', ax=ax, color='coral')
    ax.set_title('Top-k별 평균 Precision')
    ax.set_xlabel('Top-k')
    ax.set_ylabel('Precision')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    pivot_f1 = results_df.pivot_table(values='f1', index='chunk_size', columns='top_k', aggfunc='mean')
    sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
    ax.set_title('F1 Score Heatmap (Chunk Size vs Top-k)')

    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'analysis_{category_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"시각화 저장: {plot_path}")

    plt.close()


def compare_categories(results_hairstyle: pd.DataFrame, results_haircolor: pd.DataFrame, output_dir: str):
    print(f"\n{'='*70}")
    print(" 카테고리 간 비교")
    print(f"{'='*70}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    data_recall = pd.DataFrame({
        'Hairstyle': results_hairstyle.groupby('top_k')['recall'].mean(),
        'Haircolor': results_haircolor.groupby('top_k')['recall'].mean()
    })
    data_recall.plot(kind='line', marker='o', ax=ax)
    ax.set_title('카테고리별 Recall 비교')
    ax.set_xlabel('Top-k')
    ax.set_ylabel('Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    data_precision = pd.DataFrame({
        'Hairstyle': results_hairstyle.groupby('top_k')['precision'].mean(),
        'Haircolor': results_haircolor.groupby('top_k')['precision'].mean()
    })
    data_precision.plot(kind='line', marker='o', ax=ax)
    ax.set_title('카테고리별 Precision 비교')
    ax.set_xlabel('Top-k')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    data_f1 = pd.DataFrame({
        'Hairstyle': results_hairstyle.groupby('top_k')['f1'].mean(),
        'Haircolor': results_haircolor.groupby('top_k')['f1'].mean()
    })
    data_f1.plot(kind='line', marker='o', ax=ax)
    ax.set_title('카테고리별 F1 Score 비교')
    ax.set_xlabel('Top-k')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'comparison_categories.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"비교 시각화 저장: {plot_path}")

    plt.close()


def main():

    INPUT_DIR = "./evaluation_results"
    OUTPUT_DIR = "./evaluation_results"

    print("="*70)
    print(" 평가 결과 분석 및 시각화")
    print("="*70)

    hairstyle_path = os.path.join(INPUT_DIR, 'results_hairstyle.csv')
    if os.path.exists(hairstyle_path):
        results_hairstyle = pd.read_csv(hairstyle_path)
        analyze_results(results_hairstyle, 'hairstyle', OUTPUT_DIR)
    else:
        print(f"파일 없음: {hairstyle_path}")
        results_hairstyle = None

    haircolor_path = os.path.join(INPUT_DIR, 'results_haircolor.csv')
    if os.path.exists(haircolor_path):
        results_haircolor = pd.read_csv(haircolor_path)
        analyze_results(results_haircolor, 'haircolor', OUTPUT_DIR)
    else:
        print(f"파일 없음: {haircolor_path}")
        results_haircolor = None

    if results_hairstyle is not None and results_haircolor is not None:
        compare_categories(results_hairstyle, results_haircolor, OUTPUT_DIR)

    print(f"\n{'='*70}")
    print(" 분석 완료!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
