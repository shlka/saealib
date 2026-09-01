---
primary_layer: layer2
related_layers: []
page_type: entry
---

# アルゴリズム

文献上の名前がついたアルゴリズムを、saealibのコンポーネントの組み合わせとしてどう再現するかをまとめたページです。
各ページは、そのアルゴリズムの理論的な概要（saealibに依存しない一般的な説明）と、saealibでの構成方法（コンポーネントの組み合わせとPythonコード）の2部構成です。
出典の完全な書誌情報は[文献リファレンス](../references.md)にまとめてあり、このセクションの各ページはそこへリンクします。

構成が理論定義と完全一致しない場合は、獲得関数の制約、選択やarchiveの扱い、評価順序など、実装上の差分を各ページで明示します。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} EGO法
:link: ego
:link-type: doc

Gaussian Process回帰と期待改善量による逐次モデルベース最適化。
:::

:::{grid-item-card} GP-UCB法
:link: gp_ucb
:link-type: doc

Gaussian Process回帰と上側信頼限界による逐次モデルベース最適化。
:::

:::{grid-item-card} MaxUnc法
:link: maxunc
:link-type: doc

Gaussian Process回帰の予測不確実性のみを基準にする、探索専用の逐次モデルベース最適化。
:::

:::{grid-item-card} CORS-RBF法
:link: rbf_cors
:link-type: doc

RBF補間によるサロゲートモデルと、既存評価点からの距離制約による逐次モデルベース最適化。
:::

:::{grid-item-card} NSGA-II法
:link: nsga2
:link-type: doc

非優越ソートと混雑度距離による多目的遺伝アルゴリズム。多目的最適化の比較演算子の基礎。
:::

:::{grid-item-card} SPEA2法
:link: spea2
:link-type: doc

支配関係に基づく強度と密度を組み合わせた適応度、固定サイズアーカイブによる多目的遺伝アルゴリズム。
:::

:::{grid-item-card} NSGA-III法
:link: nsga3
:link-type: doc

reference-directionによるニッチ保持を用いる多目的遺伝的アルゴリズム。
many-objective optimizationを主対象とし、3目的でもreference-directionの挙動を観察できます。
:::

:::{grid-item-card} SMS-EMOA法
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
