---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# グラフ（Graph）

Componentの契約を実行順序とデータ関係へ配置すると、`ComponentGraph`になります。
`ComponentGraph`は、その配置を保持する不変のグラフです。
契約の木構造を保持するComponentContractと、ノード間の実行関係を保持するComponentGraphは別の構造です。

## ノードと参照

`ComponentNode`はComponentインスタンス、component ID、role、解決済みサービス、コンパイル用契約スナップショットを保持します。
`NodeRef`はcomponent IDと任意のroleでノードを参照し、エッジやStateBindingの端点を正規化します。
同じComponentを異なるroleで配置する場合も、NodeRefによって接続先を区別できます。

## エッジと状態

`DataEdge`はsource portからtarget portへのデータ接続です。
`ControlEdge`はデータを渡さず、sourceの完了がtargetの実行に先行する制御依存です。
`StateBinding`はノードを実際の型付きStateKeyへ結び付け、契約のstate宣言を実行時の状態へ対応付けます。

Graphのノード、エッジ、state binding、entry pointはそれぞれ値として保持され、未知のノードを参照する端点などはwell-formedness診断になります。

## 構造化領域（Structured Region）

`StructuredRegion`は、Sequence、Repeat、Loop、Branchの入れ子構造と、その領域が読む状態効果を保持します。
Loopは通常のGraph cycleとして表さず、条件と領域効果をCompilerが検証できる構造として保持します。
構造化領域のlowering後も、ExecutablePlanへ渡す実行木と状態効果の対応を保ちます。

## 生成時点と検証

GraphはPipelineやGraph builderがComponentを配置するときに生成されます。
Compilerは契約スナップショットを読み、entry point、エッジ端点、ポート互換性、状態binding、構造化領域を検証します。
Compiler ruleがGraphを解決段階で変更する場合は、変更箇所をclaimとして宣言し、未申告の変更や競合をDiagnosticsにします。

## 最小例と拡張

```python
from saealib.core import ComponentContract, ComponentGraph
from saealib.core.compiler import ComponentNode


class Empty:
    def contract(self) -> ComponentContract:
        return ComponentContract()


node = ComponentNode(component_id="root", component=Empty())
graph = ComponentGraph(nodes=(node,), entry_points=("root",))
```

この例は一つのノードを入口にしたGraphを作ります。
実際のポートとエッジの構築は、公開APIの構成に合わせて[APIリファレンス](../api/index.md)で確認してください。
Graph接続のルールを追加する場合は[フレームワーク拡張](extensions.md)を参照してください。
