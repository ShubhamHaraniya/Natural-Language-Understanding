import os
import matplotlib.pyplot as plt
import numpy as np


def load_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def novelty_rate(generated, training_set):
    """names the model invented that weren't in training data"""
    training_lower = {n.lower() for n in training_set}
    novel = [n for n in generated if n.lower() not in training_lower]
    return novel, len(novel) / len(generated) * 100 if generated else 0


def diversity(generated):
    """unique names / total — a repetitive model scores low"""
    if not generated:
        return 0
    return len(set(n.lower() for n in generated)) / len(generated) * 100


def main():
    out_dir = "evaluation_results"
    os.makedirs(out_dir, exist_ok=True)

    training = load_lines("TrainingNames.txt")
    if not training:
        print("TrainingNames.txt not found.")
        return

    models = [
        ("Vanilla RNN",    "vanilla_rnn"),
        ("BLSTM",          "blstm"),
        ("RNN + Attention","rnn_attention"),
    ]

    names_list     = []
    novelty_vals   = []
    diversity_vals = []
    novel_names    = {}
    all_gen        = {}

    print("=" * 65)
    print("QUANTITATIVE EVALUATION")
    print("=" * 65)
    print(f"\n{'Model':<22} {'Novelty %':>12} {'Diversity %':>14} {'Count':>8}")
    print("-" * 60)

    for display_name, file_name in models:
        gen = load_lines(f"generated_samples/{file_name}_names.txt")
        if not gen:
            print(f"{display_name:<22} {'(no data)':>12}")
            continue

        novel, nov_pct = novelty_rate(gen, training)
        div_pct        = diversity(gen)
        names_list.append(display_name)
        novelty_vals.append(nov_pct)
        diversity_vals.append(div_pct)
        novel_names[display_name] = novel
        all_gen[display_name]     = gen
        print(f"{display_name:<22} {nov_pct:>11.1f}% {div_pct:>13.1f}% {len(gen):>8}")

    # save novel names per model for manual inspection
    for display_name, file_name in models:
        if display_name not in novel_names:
            continue
        path = os.path.join(out_dir, f"{file_name}_novel_names.txt")
        with open(path, "w", encoding="utf-8") as f:
            for n in novel_names[display_name]:
                f.write(n + "\n")
        print(f"\nNovel names for {display_name} → {path}")
        print(f"  ({len(novel_names[display_name])} names): {', '.join(novel_names[display_name])}")

    # plot 1: novelty vs diversity bar chart
    x     = np.arange(len(names_list))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, novelty_vals,  width, label='Novelty %',  color='#4C72B0')
    bars2 = ax.bar(x + width/2, diversity_vals, width, label='Diversity %', color='#55A868')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Novelty Rate & Diversity Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names_list)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "novelty_diversity_comparison.png"), dpi=150)
    print(f"\nSaved: novelty_diversity_comparison.png")

    # plot 2: name length distributions
    colors = ['#4C72B0', '#55A868', '#C44E52']
    fig, axes = plt.subplots(1, len(names_list), figsize=(5 * len(names_list), 4), sharey=True)
    for i, (name, color) in enumerate(zip(names_list, colors)):
        if name not in all_gen:
            continue
        lengths = [len(n) for n in all_gen[name]]
        axes[i].hist(lengths, bins=range(1, max(lengths) + 2), color=color, edgecolor='white', alpha=0.85)
        axes[i].set_title(name)
        axes[i].set_xlabel('Name Length')
        if i == 0:
            axes[i].set_ylabel('Count')
        axes[i].axvline(np.mean(lengths), color='black', linestyle='--', linewidth=1,
                        label=f'Mean: {np.mean(lengths):.1f}')
        axes[i].legend(fontsize=8)
    plt.suptitle('Name Length Distribution', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "name_length_distribution.png"), dpi=150)
    print("Saved: name_length_distribution.png")

    # plot 3: novel vs copied pie charts
    fig, axes = plt.subplots(1, len(names_list), figsize=(5 * len(names_list), 4))
    for i, name in enumerate(names_list):
        if name not in all_gen:
            continue
        n_novel  = len(novel_names[name])
        n_copied = len(all_gen[name]) - n_novel
        axes[i].pie([n_novel, n_copied], labels=['Novel', 'From Training'],
                    autopct='%1.0f%%', colors=['#55A868', '#C44E52'], startangle=90)
        axes[i].set_title(name)
    plt.suptitle('Novel vs Training-Copied Names', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "novel_vs_copied.png"), dpi=150)
    print("Saved: novel_vs_copied.png")

    # plot 4: character frequency — which chars does each model lean on?
    fig, axes = plt.subplots(1, len(names_list), figsize=(5 * len(names_list), 5))
    for i, name in enumerate(names_list):
        if name not in all_gen:
            continue
        text  = ''.join(all_gen[name]).lower()
        chars = sorted(set(text))
        freqs = [text.count(c) for c in chars]
        axes[i].barh(chars, freqs, color=colors[i], edgecolor='white')
        axes[i].set_title(name)
        axes[i].set_xlabel('Frequency')
        axes[i].tick_params(axis='y', labelsize=7)
    plt.suptitle('Character Frequency in Generated Names', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "char_frequency.png"), dpi=150)
    print("Saved: char_frequency.png")

    # qualitative breakdown per model
    print("\n" + "=" * 65)
    print("QUALITATIVE ANALYSIS")
    print("=" * 65)

    for display_name, file_name in models:
        if display_name not in all_gen:
            continue
        gen            = all_gen[display_name]
        training_lower = {n.lower() for n in training}

        print(f"\n--- {display_name} ---")
        print(f"  Samples: {', '.join(gen[:20])}")

        short     = [n for n in gen if len(n) <= 1]
        long_ones = [n for n in gen if len(n) > 15]
        repeated  = [n for n in gen if any(c * 3 in n.lower() for c in 'abcdefghijklmnopqrstuvwxyz')]
        copies    = [n for n in gen if n.lower() in training_lower]

        print(f"  Exact copies: {len(copies)},  Novel: {len(gen) - len(copies)}")
        print(f"  Very short (<=1 char): {len(short)},  Very long (>15 chars): {len(long_ones)}")
        print(f"  Repeated chars (e.g. 'aaa'): {len(repeated)}")

        if short:     print(f"    Short examples: {short[:5]}")
        if long_ones: print(f"    Long examples: {long_ones[:5]}")
        if repeated:  print(f"    Repeated examples: {repeated[:5]}")

    print("\n" + "=" * 65)
    print("DISCUSSION")
    print("=" * 65)
    print("""
Realism:
  Vanilla RNN captures basic phonetic patterns but struggles with
  longer names — vanishing gradients make it hard to stay consistent.

  BLSTM produces the cleanest names. The cell state remembers what
  characters came before, which keeps the output structurally valid.

  RNN + Attention attends over its own generation history, which helps
  a little, but for names as short as 6-7 characters on average,
  there isn't much history to attend over — so the gains are modest.

Common Failure Modes:
  1. Vanilla RNN — corrupted names due to character-level instability
  2. BLSTM — over-memorization, most outputs are just training names
  3. Attention — same memorization issue, marginal improvement in novelty
  4. All models — higher novelty often means more noise, not real creativity
""")
    print(f"All results saved to {out_dir}/")


if __name__ == "__main__":
    main()
