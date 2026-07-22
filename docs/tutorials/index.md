# チュートリアル

よくある最適化シナリオに対応する、状況別のセットアップガイドです。高レベルAPIから、手動で組み立てる`Optimizer`まで扱います。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 単目的最適化
:link: single_objective
:link-type: doc

評価コストの高い単目的問題を、`minimize()`から手動で組み立てる`Optimizer`まで、段階的に解きます。
:::

:::{grid-item-card} 多目的最適化
:link: multi_objective
:link-type: doc

目的間にトレードオフのある問題を解き、パレートフロントを抽出します。
:::

:::{grid-item-card} 制約付き最適化
:link: constraints
:link-type: doc

不等式制約を定義し、実行不可能解の扱い方を制御します。
:::

:::{grid-item-card} 混合変数最適化
:link: mixed_variable
:link-type: doc

連続変数に加えて整数変数、カテゴリカル変数を含む問題を解きます。
:::

:::{grid-item-card} 動的な切り替え
:link: dynamic_optimization
:link-type: doc

サロゲートの予測精度に応じて、評価戦略やサロゲートマネージャーを実行中に切り替えます。
:::

:::{grid-item-card} 再現性とチェックポイント
:link: checkpoint
:link-type: doc

長時間実行する最適化を再現可能・再開可能にします。
:::

:::{grid-item-card} 進捗のログ記録
:link: logging
:link-type: doc

標準の`logging`モジュールで、最適化の進捗を記録します。
:::

:::{grid-item-card} 外部ライブラリとの連携
:link: external_libraries
:link-type: doc

scikit-learnなど外部ライブラリのモデルを、サロゲートとして組み込みます。
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
