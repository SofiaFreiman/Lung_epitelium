import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
from typing import Optional, List, Tuple


def significance(x: pd.Series, pval_cut: float = 0.001, fc_cut: float = 0) -> str:
    """
    Classify genes based on adjusted p-value and fold change.
    """
    pval, logfc = x[0], x[1]
    return 'changed' if abs(logfc) > fc_cut and pval < pval_cut else 'unchanged'


def dot_color(x: pd.Series) -> str:
    """
    Assign color based on gene significance and direction of change.
    """
    significance, logfc = x[0], x[1]
    if significance == 'changed':
        return 'red' if logfc > 0 else 'blue'
    return 'gray'


def volcano(
    df: pd.DataFrame,
    text_pval_cut: float = 20,
    text_fc_cut: float = 1,
    log_method: str = "natural",
    title: str = "Volcano Plot",
    xlim: Optional[float] = None,
    ylim: Optional[float] = None,
    print_gene_from_list: bool = True,
    genes_of_interest: Optional[List[str]] = None,
    print_sig_genes: bool = True,
    gene_fontsize: int = 10,
    figsize: Tuple[int, int] = (6, 5),
    linecolor: str = 'darkgrey',
    too_crowded = False,
    color_left: str = 'royalblue',
    color_right: str = 'orangered',
) -> None:
    """
    Create a volcano plot from a dataframe with 'pvals_adj' and 'logfoldchanges'.

    Args:
        df (pd.DataFrame): DataFrame with columns 'pvals_adj', 'logfoldchanges'.
        text_pval_cut (float): Threshold for -logP to label genes.
        text_fc_cut (float): Threshold for abs(logFC) to label genes.
        title (str): Plot title.
        xlim (float or None): Set max x-axis limit.
        ylim (float or None): Set max y-axis limit.
        print_gene_from_list (bool): Annotate genes from provided list.
        genes_of_interest (list or None): List of gene names (index) to annotate.
        print_sig_genes (bool): Annotate top significant genes automatically.
        gene_fontsize (int): Font size for annotations.
        figsize (tuple): Size of the figure.
        linecolor (str): Color of label lines.
        color_left (str): Color for the downregulated genes.
        color_right (str): Color for the upregulated genes.
    """

    df = df.copy()
    # Set log function and y-axis label based on user choice
    if log_method == "natural":
        log_func = np.log
        log_label = "-ln(p-value)"
    elif log_method == "log10":
        log_func = np.log10
        log_label = "-log10(p-value)"
    else:
        raise ValueError("log_method must be either 'natural' or 'log10'")
    df['-logP'] = -log_func(df['pvals_adj'])

    # Replace inf values in -logP
    inf_mask = np.isinf(df['-logP'])
    if inf_mask.any():
        max_val = df.loc[~inf_mask, '-logP'].max()
        df.loc[inf_mask, '-logP'] = max_val

    # Classify significance
    df['significance'] = df[['pvals_adj', 'logfoldchanges']].apply(significance, axis=1)

    # Assign colors
    df['dot_color'] = df[['significance', 'logfoldchanges']].apply(dot_color, axis=1)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, facecolor=(1, 1, 1, 0))
    sns.scatterplot(
        x="logfoldchanges", y="-logP", data=df,
        hue="dot_color", palette={"gray": 'gray', "red": color_right, "blue": color_left},
        s=10, linewidth=0.2, ax=ax, legend=False,
    )
    if too_crowded:
      sns.scatterplot(
          x="logfoldchanges", y="-logP", data=df.sample(len(df)//10),
          hue="dot_color", palette={"gray": 'gray', "red": color_right, "blue": color_left},
          s=10, linewidth=0.2, ax=ax, legend=False,
          )

    # Style
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("log Fold Change", fontsize=14)
    ax.set_ylabel(log_label, fontsize=14)
    ax.tick_params(labelsize=10)
    ax.grid(False)
    if xlim:
        ax.set_xlim(-xlim, xlim)
    if ylim:
        ax.set_ylim(0, ylim)

    # Annotate significant genes
    texts = []
    if print_sig_genes:
        sig_genes = df[(df['-logP'] > text_pval_cut) & (df['logfoldchanges'].abs() >= text_fc_cut)]
        sig_genes = sig_genes[~sig_genes.index.str.contains("ENSG|Hu")]
        for gene in sig_genes.index:
            texts.append(
                plt.text(sig_genes.loc[gene, "logfoldchanges"],
                         sig_genes.loc[gene, "-logP"],
                         gene,
                         fontsize=gene_fontsize, weight='bold')
            )

    # Annotate genes of interest
    if print_gene_from_list and genes_of_interest:
        goi = df[df.index.isin(genes_of_interest)].copy()
        if print_sig_genes and not sig_genes.empty:
            goi = goi[~goi.index.isin(sig_genes.index)]

        for gene in goi.index:
            ha = 'left' if goi.loc[gene, 'logfoldchanges'] > 0 else 'right'
            texts.append(
                plt.text(goi.loc[gene, "logfoldchanges"],
                         goi.loc[gene, "-logP"],
                         gene,
                         fontsize=gene_fontsize, weight='bold', ha=ha)
            )

    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color=linecolor, lw=0.6))

    plt.tight_layout()
