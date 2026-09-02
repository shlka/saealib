---
primary_layer: cross
related_layers: [layer2, layer3]
page_type: entry
---

# サロゲートモデリング

`TrainingSet`はアーカイブまたは個体群から学習データを作り、`Surrogate`はそのデータから目的値を予測します。`SurrogateManager`はフィットと予測のタイミングを調整し、`AcquisitionFunction`は予測を候補選択用のスコアに変換します。`AccuracyBasedSurrogateSwitcher`は、予測精度の評価に応じて`SurrogateManager`または`OptimizationStrategy`を切り替えます。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`brain;sd-mr-1` サロゲート
:link: surrogate_modeling/surrogate
:link-type: doc
学習データからモデルをフィットし、候補の予測を返します。
:::
:::{grid-item-card} {fa}`sitemap;sd-mr-1` サロゲートマネージャー
:link: surrogate_modeling/surrogate_manager
:link-type: doc
モデルのフィットとバッチ予測を調整します。
:::
:::{grid-item-card} {fa}`database;sd-mr-1` 学習データ集合
:link: surrogate_modeling/training_set
:link-type: doc
ArchiveまたはPopulationから学習用の入力とラベルを構築します。
:::
:::{grid-item-card} {fa}`calculator;sd-mr-1` 獲得関数
:link: surrogate_modeling/acquisition_functions
:link-type: doc
予測結果を候補選択用のスカラー値に変換します。
:::
:::{grid-item-card} {fa}`toggle-on;sd-mr-1` 精度ベースのサロゲート切り替え
:link: surrogate_modeling/surrogate_switching
:link-type: doc
サロゲートの精度を評価し、実行中に使う構成を切り替えます。
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
