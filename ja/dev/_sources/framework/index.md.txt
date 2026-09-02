---
primary_layer: layer4
related_layers: [layer3]
page_type: entry
---

# Framework

saealibのフレームワークはComponentの契約をComponentGraphに配置し、Compilerが検証済みのExecutablePlanへ変換するための実行基盤を提供します。
契約構造と実行フローは別の関係として扱います。

## 概念ページ

Componentは`contract()`で静的な契約を返します。
ComponentContractは、Componentが保持するコンポーネントの契約包含と必要な能力を記述するもので、Componentの継承階層を表すものではありません。
`PortSpec`、`DataSpec`、`StateContract`は契約を構成する宣言的な値であり、Componentサブクラスではありません。

次のページで、拡張時に必要な概念を分担して説明します。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`puzzle-piece;sd-mr-1` Component
:link: component
:link-type: doc
Component Protocol、`contract()`、PartSpecの関係。
:::

:::{grid-item-card} {fa}`file-signature;sd-mr-1` ComponentContract
:link: contract
:link-type: doc
契約の構成要素と不変条件。
:::

:::{grid-item-card} {fa}`plug;sd-mr-1` Specs
:link: specs
:link-type: doc
ポート、データ、サービス、互換性を表す宣言的な値
:::

:::{grid-item-card} {fa}`diagram-project;sd-mr-1` Graph
:link: graph
:link-type: doc
ノード、エッジ、状態バインディング、構造化領域。
:::

:::{grid-item-card} {fa}`gears;sd-mr-1` Compiler
:link: compiler
:link-type: doc
検証、解決、診断、ExecutablePlan。
:::

:::{grid-item-card} {fa}`microchip;sd-mr-1` Runtime
:link: runtime
:link-type: doc
ExecutablePlanの実行、状態適用、再開、非同期待機。
:::

:::{grid-item-card} {fa}`square-root-variable;sd-mr-1` SearchSpace
:link: ../concepts/problem_and_ranking/search_space
:link-type: doc
候補表現、サービス、RepresentationSpecの境界。
:::

:::{grid-item-card} {fa}`comment-dots;sd-mr-1` Feedback
:link: ../concepts/observation_and_state/feedback
:link-type: doc
候補ID、観測、真値と予測の対応。
:::

:::{grid-item-card} {fa}`database;sd-mr-1` OptimizationState
:link: ../concepts/observation_and_state/optimization_state
:link-type: doc
Stage互換状態とgraph-native状態の所有境界。
:::

::::

## 実行フロー

Runtime上の関係は`Component → ComponentNode → ComponentGraph → Compiler → ExecutablePlan → ExecutionRuntime`です。
この流れは契約ツリーとは別であり、ComponentContractが実行順序を直接表すことはありません。

詳細な状態境界は[OptimizationState](../concepts/observation_and_state/optimization_state.md)と[Runtime](runtime.md)で、候補表現と観測の境界は[SearchSpace](../concepts/problem_and_ranking/search_space.md)と[Feedback](../concepts/observation_and_state/feedback.md)で説明します。

## 拡張経路

既存のビルトインコンポーネントの振る舞いだけを差し替える場合は、対応する拡張ページを選びます。
新しい契約、候補表現、Graph接続、Compilerルール、Runtimeの意味論が必要な場合は[フレームワーク拡張](extensions.md)を参照してください。
公開型のimport経路は[APIリファレンス](../api/index.md)で確認します。

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
