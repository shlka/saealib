---
primary_layer: cross
related_layers: [layer2, layer3]
page_type: entry
---

# 観測と状態

CallbackManagerは実行境界でイベントを発行し、外部コードが進行状況と結果を観測できるようにします。Stageは既存の世代処理と`OptimizationState`を受け渡す互換性境界であり、graph-nativeコンポーネントの状態境界とは異なります。`OptimizationState`は実行値と再開に必要な値を保持し、`Population`は個体、目的値、制約値などの個体群データを管理します。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`eye;sd-mr-1` CallbackManager
:link: observation_and_state/callbacks
:link-type: doc
イベントを記録し、実行中の処理を観測します。
:::

:::{grid-item-card} {fa}`circle-info;sd-mr-1` 診断と観測
:link: observation_and_state/diagnostics
:link-type: doc
エラー、警告、ログ、履歴、コールバック、コンパイラ診断に適した仕組みを選びます。
:::

:::{grid-item-card} {fa}`comment-dots;sd-mr-1` フィードバック
:link: observation_and_state/feedback
:link-type: doc
候補と観測を対応付け、アルゴリズムが利用できる形へ渡します。
:::

:::{grid-item-card} {fa}`bars-staggered;sd-mr-1` Stage
:link: observation_and_state/stage
:link-type: doc
世代処理を構成する互換用の実行単位です。
:::

:::{grid-item-card} {fa}`hard-drive;sd-mr-1` OptimizationState
:link: observation_and_state/optimization_state
:link-type: doc
実行状態、結果、チェックポイントに必要な値を保持します。
:::

:::{grid-item-card} {fa}`users;sd-mr-1` Population
:link: observation_and_state/population
:link-type: doc
個体と評価結果を集団およびArchiveとして管理します。
:::

::::

```{toctree}
:hidden:

observation_and_state/callbacks
observation_and_state/diagnostics
observation_and_state/feedback
observation_and_state/stage
observation_and_state/optimization_state
observation_and_state/population
```
