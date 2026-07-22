# アルゴリズム

文献上の名前がついたアルゴリズムを、saealibのコンポーネントの組み合わせとしてどう再現するかをまとめたページです。
各ページは、そのアルゴリズムの理論的な概要(saealibに依存しない一般的な説明)と、saealibでの構成方法(コンポーネントの組み合わせとPythonコード)の2部構成です。
出典の完全な書誌情報は[文献リファレンス](../references.md)にまとめてあり、このセクションの各ページはそこへリンクします。

構成が理論上の定義と厳密に一致しない場合(近似)は、各ページにその旨を明記します。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} EGO
:link: ego
:link-type: doc

Gaussian Process回帰と期待改善量(Expected Improvement)による逐次モデルベース最適化。
:::

:::{grid-item-card} GP-UCB
:link: gp_ucb
:link-type: doc

Gaussian Process回帰と上側信頼限界(Upper Confidence Bound)による逐次モデルベース最適化。
:::

:::{grid-item-card} MaxUnc
:link: maxunc
:link-type: doc

Gaussian Process回帰の予測不確実性のみを基準にする、探索専用の逐次モデルベース最適化。
:::

:::{grid-item-card} CORS-RBF
:link: rbf_cors
:link-type: doc

RBF補間による代理モデルと、既存評価点からの距離制約による逐次モデルベース最適化。
:::

:::{grid-item-card} NSGA-II
:link: nsga2
:link-type: doc

非優越ソートと混雑度距離による多目的遺伝アルゴリズム。多目的最適化の比較演算子の基礎。
:::

:::{grid-item-card} SPEA2
:link: spea2
:link-type: doc

支配関係に基づく強度と密度を組み合わせた適応度、固定サイズアーカイブによる多目的遺伝アルゴリズム。
:::

:::{grid-item-card} NSGA-III
:link: nsga3
:link-type: doc

参照点ベースのニッチ保存による多目的遺伝アルゴリズム。四つ以上の目的を持つmany-objective最適化向け。
:::

:::{grid-item-card} SMS-EMOA
:link: sms_emoa
:link-type: doc

被支配超体積を選択基準に直接組み込んだ、定常状態の多目的進化アルゴリズム。
:::

::::

```{toctree}
:hidden:

ego
gp_ucb
maxunc
rbf_cors
nsga2
spea2
nsga3
sms_emoa
```
