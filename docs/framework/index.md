---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# フレームワーク

saealibのフレームワークは、Componentの契約をComponentGraphへ配置し、Compilerが検証済みのExecutablePlanへ変換するための実行基盤です。
契約の構造と実行の流れは別の関係として扱います。

## 概念ページ

Componentは`contract()`で自身の静的な契約を返します。
ComponentContractはComponentの継承階層ではなく、Componentが保持するコンポーネントと要求する能力を記述する契約の包含関係です。
PortSpec、DataSpec、StateContractなどはComponentの派生型ではなく、契約を構成する宣言値です。

次のページで、拡張時に必要な概念を分担して説明します。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`puzzle-piece;sd-mr-1` コンポーネント（Component）
:link: component
:link-type: doc
Component Protocol、`contract()`、PartSpecの関係。
:::

:::{grid-item-card} {fa}`file-signature;sd-mr-1` コンポーネント契約（ComponentContract）
:link: contract
:link-type: doc
契約の構成要素と不変条件。
:::

:::{grid-item-card} {fa}`plug;sd-mr-1` 宣言要素（Specs）
:link: specs
:link-type: doc
ポート、データ、サービス、互換性の宣言値。
:::

:::{grid-item-card} {fa}`diagram-project;sd-mr-1` グラフ（Graph）
:link: graph
:link-type: doc
ノード、エッジ、状態バインディング、構造化領域。
:::

:::{grid-item-card} {fa}`gears;sd-mr-1` コンパイラ（Compiler）
:link: compiler
:link-type: doc
検証、解決、診断、ExecutablePlan。
:::

:::{grid-item-card} {fa}`microchip;sd-mr-1` ランタイム（Runtime）
:link: runtime
:link-type: doc
ExecutablePlanの実行、状態適用、再開、非同期待機。
:::

:::{grid-item-card} {fa}`square-root-variable;sd-mr-1` 探索空間（SearchSpace）
:link: ../concepts/problem_and_ranking/search_space
:link-type: doc
候補表現、サービス、RepresentationSpecの境界。
:::

:::{grid-item-card} {fa}`comment-dots;sd-mr-1` フィードバック（Feedback）
:link: ../concepts/observation_and_state/feedback
:link-type: doc
候補ID、観測、真値と予測の対応。
:::

:::{grid-item-card} {fa}`database;sd-mr-1` 最適化状態（OptimizationState）
:link: ../concepts/observation_and_state/optimization_state
:link-type: doc
Stage互換状態とgraph-native状態の所有境界。
:::

::::

## 実行の流れ

実行時の関係は、`Component → ComponentNode → ComponentGraph → Compiler → ExecutablePlan → ExecutionRuntime`です。
この流れは契約の木構造とは別であり、ComponentContractがそのまま実行順序を表すわけではありません。

詳細な状態境界は[OptimizationState](../concepts/observation_and_state/optimization_state.md)と[Runtime](runtime.md)で、候補表現と観測の境界は[探索空間（SearchSpace）](../concepts/problem_and_ranking/search_space.md)と[Feedback](../concepts/observation_and_state/feedback.md)で説明します。

## 拡張方法

既存のビルトインコンポーネントの振る舞いだけを差し替える場合は、対応する拡張ページを選びます。
新しい契約、候補表現、Graph接続、Compiler rule、Runtime意味論が必要な場合は[フレームワーク拡張](extensions.md)を参照してください。
公開されている型のimport経路は[APIリファレンス](../api/index.md)で確認してください。

```{toctree}
:hidden:

component
contract
specs
graph
compiler
runtime
extensions
```
