---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# コンポーネント契約（ComponentContract）

Componentが要求する能力と提供する境界は、`ComponentContract`という一つの宣言値にまとめます。
Componentとの関係は継承ではなく、Componentが返す契約の包含です。

## 構成要素

| フィールド | 宣言する内容 |
|---|---|
| `ports` | roleごとの`PortContract` |
| `required_services` | Component全体が必要とする`ServiceRequirement` |
| `parts` | 保持するコンポーネントの`PartSpec` |
| `lifecycle` | `events`とFeedbackの境界 |
| `state` | `reads`、`writes`、`exports`、`reads_enumerable` |
| `execution` | `required_runtime_capabilities`と`offered_runtime_capabilities` |
| `assumptions` | コンパイラが扱う前提条件 |

`ports`のroleは、同じComponentをGraph上の役割ごとに接続するための名前付き集合です。
`parts`は子Componentの実装を複製する値ではなく、子の契約を親の契約へ含める宣言です。

## 各契約が持つ情報

| 契約 | 保持する情報 | 主な検証 |
|---|---|---|
| `PortContract` | 入力と出力の`PortSpec` | ポート名、方向、DataSpec、cardinality |
| `StateContract` | 読み取り、書き込み、公開する`StateKey` | 宣言外の状態アクセス、状態効果 |
| `LifecycleContract` | 消費するイベントとFeedbackContract | イベントとFeedbackの互換性 |
| `ExecutionContract` | 必要または提供するRuntime capability | 実行環境の能力不足 |
| `AssumptionSet` | コンポーネントが置く前提条件 | 前提条件の登録と既定値 |

## 不変条件

契約は凍結されたデータクラスとして扱われ、Compilerの検証中に内容を変更しません。
`ports`、`required_services`、`parts`は対応する宣言値だけを含み、role名やコンポーネント名は識別子として検証されます。
状態の`reads`、`writes`、`exports`は型付きStateKeyで表し、Runtimeは宣言外のキーをComponentへ公開しません。
実行能力の要求がCompileContextの提供能力に含まれない場合、CompilerはExecutablePlanを実行可能として確定できません。

## 生成時点と利用時点

Componentは`contract()`で契約を生成し、ComponentNodeはコンパイル単位のスナップショットを保持します。
Compilerは契約をポート互換性、サービス解決、状態効果、ライフサイクル、Runtime capabilityの検証に使います。
Runtimeは検証済みのExecutablePlanに含まれる契約境界に従ってStateViewを作り、結果のStatePatchを適用します。

## 最小例

```python
from saealib.core import ComponentContract, StateContract


def contract() -> ComponentContract:
    return ComponentContract(state=StateContract())
```

ポートの宣言値は[宣言要素（Specs）](specs.md)を、Graphへの配置は[Graph](graph.md)を参照してください。

## 代表的なDiagnostics

契約の型が誤っている場合は契約生成時に`ValidationError`になります。
接続先のポート、要求サービス、状態効果、ライフサイクル、Runtime capabilityが整合しない場合はCompilerの`Diagnostic`として報告されます。
Compiler ruleの公開経路はリリースごとに異なる可能性があるため、具体的なimportは[APIリファレンス](../api/index.md)で確認してください。
