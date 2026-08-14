---
primary_layer: cross
related_layers: []
page_type: entry
---

# Tutorials

具体的な最適化シナリオに対する実装例を示します。
単目的、多目的、制約、混合変数など、目的に合うページを選べます。
シナリオごとの実装例であり、構成の深さを前提にしません。

## 最適化シナリオ

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`bullseye;sd-mr-1` Single-Objective Optimization
:link: single_objective
:link-type: doc

Solve an expensive single-objective problem step by step, from `minimize()` to manually assembling an `Optimizer`.
:::

:::{grid-item-card} {fa}`layer-group;sd-mr-1` Multi-Objective Optimization
:link: multi_objective
:link-type: doc

Solve a problem with trade-offs between objectives and extract the Pareto front.
:::

:::{grid-item-card} {fa}`shield-halved;sd-mr-1` Constrained Optimization
:link: constraints
:link-type: doc

Define inequality constraints and control how infeasible solutions are handled.
:::

:::{grid-item-card} {fa}`shapes;sd-mr-1` Mixed-Variable Optimization
:link: mixed_variable
:link-type: doc

Solve problems that include integer and categorical variables alongside continuous variables.
:::

:::{grid-item-card} {fa}`arrows-rotate;sd-mr-1` Dynamic Switching
:link: dynamic_optimization
:link-type: doc

Switch the evaluation strategy or `SurrogateManager` at runtime based on the surrogate's prediction accuracy.
:::

:::{grid-item-card} {fa}`floppy-disk;sd-mr-1` Reproducibility and Checkpointing
:link: checkpoint
:link-type: doc

Make long-running optimizations reproducible and resumable.
:::

:::{grid-item-card} {fa}`file-lines;sd-mr-1` Logging Progress
:link: logging
:link-type: doc

Record optimization progress with the standard `logging` module.
:::

:::{grid-item-card} {fa}`plug;sd-mr-1` Integrating External Libraries
:link: external_libraries
:link-type: doc

Incorporate models from external libraries such as scikit-learn as surrogates.
:::

::::

```{toctree}
:hidden:

single_objective
multi_objective
constraints
mixed_variable
dynamic_optimization
checkpoint
logging
external_libraries
```

## 構成と操作

ビルトインコンポーネントの選択、低レベルAPIの構成、実行経路へのHookやCallbackの追加を扱います。
詳細な内容は各ガイドで扱い、ここでは概要とリンクを示します。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`puzzle-piece;sd-mr-1` ビルトインコンポーネントを選んで差し替える
:link: component_swap
:link-type: doc
責務に対応するビルトインコンポーネントを選びます。
:::

:::{grid-item-card} {fa}`sliders;sd-mr-1` Optimizerを低レベルAPIで組み立てる
:link: lowlevel_api
:link-type: doc
Optimizerの構成と実行を確認します。
:::

:::{grid-item-card} {fa}`eye;sd-mr-1` Hook、Stage、Callbackを選ぶ
:link: interface_hooks
:link-type: doc
実行経路への追加と観測を選びます。
:::

:::{grid-item-card} {fa}`rocket;sd-mr-1` 高レベルAPIで最適化を実行する
:link: highlevel_api
:link-type: doc
高レベルAPIを使った最適化の実行とResultの読み方を確認します。
:::

::::

```{toctree}
:hidden:

component_swap
lowlevel_api
interface_hooks
highlevel_api
custom_components
```
