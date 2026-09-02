---
primary_layer: cross
related_layers: []
page_type: entry
---

# チュートリアル

具体的な最適化シナリオに対する実装例を示します。
単目的、多目的、制約、混合変数など、目的に合うページを選べます。
シナリオごとの実装例であり、構成の深さを前提にしません。

## 最適化シナリオ

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`bullseye;sd-mr-1` 単目的最適化
:link: single_objective
:link-type: doc

アルゴリズム、サロゲートモデル、評価戦略を選び、評価コストの高い単目的問題を`minimize()`で解きます。
:::

:::{grid-item-card} {fa}`layer-group;sd-mr-1` 多目的最適化
:link: multi_objective
:link-type: doc

目的間にトレードオフのある問題を解き、パレートフロントを抽出します。
:::

:::{grid-item-card} {fa}`shield-halved;sd-mr-1` 制約付き最適化
:link: constraints
:link-type: doc

不等式制約を定義し、実行不可能解の扱い方を制御します。
:::

:::{grid-item-card} {fa}`shapes;sd-mr-1` 混合変数最適化
:link: mixed_variable
:link-type: doc

連続変数に加えて整数変数、カテゴリ変数を含む問題を解きます。
:::

:::{grid-item-card} {fa}`arrows-rotate;sd-mr-1` 動的な切り替え
:link: dynamic_optimization
:link-type: doc

サロゲートの予測精度に応じて、評価戦略や`SurrogateManager`を実行中に切り替えます。
:::

:::{grid-item-card} {fa}`floppy-disk;sd-mr-1` 再現性とチェックポイント
:link: checkpoint
:link-type: doc

長時間実行する最適化を再現可能かつ再開可能にします。
:::

:::{grid-item-card} {fa}`file-lines;sd-mr-1` 進捗のログ記録
:link: logging
:link-type: doc

標準の`logging`モジュールで、最適化の進捗を記録します。
:::

:::{grid-item-card} {fa}`plug;sd-mr-1` 外部ライブラリとの連携
:link: external_libraries
:link-type: doc

アダプターを通じて、scikit-learnとPyTorchのモデルをサロゲートとして、pymooの演算子、アルゴリズム、問題を組み込みます。
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

## 構成と制御

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
サロゲート込みの完全な構成、終了条件の組み合わせ、`iterate()` による世代ごとの取得を扱います。
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

:::{grid-item-card} {fa}`puzzle-piece;sd-mr-1` 独自Componentを実装する
:link: custom_components
:link-type: doc
既存の契約に沿った独自Componentを実装し、登録してテストします。外部Operatorのnative移植も扱います。
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
