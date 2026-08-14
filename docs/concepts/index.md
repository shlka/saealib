---
primary_layer: cross
related_layers: [layer2, layer3, layer4]
page_type: entry
---

# 最適化の構成要素

Component、Problem、Population、Comparatorなど、最適化を構成する要素を整理します。
責務ごとに整理したページから、ビルトインコンポーネントの詳細を参照できます。
契約から実行計画を作るフレームワークの詳細は [フレームワーク](../framework/index.md) にまとめています。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`compass;sd-mr-1` 問題定義と順位付け
:link: problem_and_ranking
:link-type: doc
Problemの定義、制約処理、順位付け、非劣解処理を扱います。
:::

:::{grid-item-card} {fa}`dna;sd-mr-1` 探索アルゴリズム
:link: search_algorithms
:link-type: doc
候補生成、変異、親の選択、次世代に残す個体の選択を扱います。
:::

:::{grid-item-card} {fa}`brain;sd-mr-1` 代理モデル
:link: surrogate_modeling
:link-type: doc
学習データ、予測、獲得関数によるスコア計算、モデルの切り替えを扱います。
:::

:::{grid-item-card} {fa}`play;sd-mr-1` 実行と評価
:link: execution_and_evaluation
:link-type: doc
初期化、候補の評価、世代処理、終了判定の流れを扱います。
:::

:::{grid-item-card} {fa}`eye;sd-mr-1` 観測と状態
:link: observation_and_state
:link-type: doc
イベント観測、Stage互換境界、実行状態、集団データを扱います。
:::

:::{grid-item-card} {fa}`wand-magic-sparkles;sd-mr-1` 拡張ガイドライン
:link: extension_guidelines
:link-type: doc
既存コンポーネントの差し替え、Hook、Stage、Callback、フレームワーク拡張の使い分けを扱います。
:::

::::

最初に最適化を実行する場合は [クイックスタート](../getting_started/quickstart.md) を、拡張点を選ぶ場合は [拡張ガイドライン](extension_guidelines.md) を参照してください。
各型の公開import経路は [APIリファレンス](../api/index.md) で確認してください。

```{toctree}
:hidden:

extension_guidelines
problem_and_ranking
search_algorithms
surrogate_modeling
execution_and_evaluation
observation_and_state
```
