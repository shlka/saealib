---
primary_layer: cross
related_layers: [layer2, layer3]
page_type: entry
---

# 代理モデル

TrainingSetはArchiveやPopulationから学習データを作り、Surrogateがそのデータで目的値を予測します。
SurrogateManagerは学習と予測のタイミングを調整し、AcquisitionFunctionは予測を候補選択用のスコアへ変換します。
AccuracyBasedSurrogateSwitcherは予測精度の評価結果に応じて、SurrogateManagerやOptimizationStrategyを切り替えます。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`brain;sd-mr-1` 代理モデル（Surrogate）
:link: surrogate_modeling/surrogate
:link-type: doc
学習データからモデルを学習し、候補の予測値を返します。
:::

:::{grid-item-card} {fa}`sitemap;sd-mr-1` 代理モデル管理（SurrogateManager）
:link: surrogate_modeling/surrogate_manager
:link-type: doc
モデルの学習とバッチ予測を調整します。
:::

:::{grid-item-card} {fa}`database;sd-mr-1` 学習データセット（TrainingSet）
:link: surrogate_modeling/training_set
:link-type: doc
ArchiveやPopulationから学習用の入力とラベルを構築します。
:::

:::{grid-item-card} {fa}`calculator;sd-mr-1` 獲得関数（AcquisitionFunction）
:link: surrogate_modeling/acquisition_functions
:link-type: doc
予測結果を候補選択に使うスカラーのスコアへ変換します。
:::

:::{grid-item-card} {fa}`toggle-on;sd-mr-1` 精度ベースの切り替え（AccuracyBasedSurrogateSwitcher）
:link: surrogate_modeling/surrogate_switching
:link-type: doc
Surrogateの精度を評価し、実行中に使う構成を切り替えます。
:::

::::

```{toctree}
:hidden:

surrogate_modeling/surrogate
surrogate_modeling/surrogate_manager
surrogate_modeling/training_set
surrogate_modeling/acquisition_functions
surrogate_modeling/surrogate_switching
```
