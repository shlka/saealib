---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# Graph

Componentの契約を実行順序とデータ関係へ配置すると、`ComponentGraph`になります。 `ComponentGraph`は、その配置を保持する不変のグラフです。 契約の木構造を保持するComponentContractと、ノード間の実行関係を保持するComponentGraphは別の構造です。

## ComponentGraphの役割

ComponentGraphは、Componentの契約を実行順序とデータ関係へ配置する境界を所有します。 契約の木構造を保持するComponentContractと、ノード間の実行関係を保持するGraphを分離し、Compilerへ自己完結した構造を渡します。

## ノードと参照

`ComponentNode`はComponentインスタンス、コンポーネントID、ロール、解決済みサービス、コンパイル用の契約スナップショットを保持します。`NodeRef`はコンポーネントIDと任意のロールでノードを参照し、エッジとStateBindingの端点を正規化します。同じComponentを異なるロールで配置した場合も、`NodeRef`によって接続先を区別できます。

## エッジと状態

`DataEdge`はsource portからtarget portへのデータ接続です。 `ControlEdge`はデータを渡さず、sourceの完了がtargetの実行に先行する制御依存です。 `StateBinding`はノードを実際の型付きStateKeyへ結び付け、契約のstate宣言を実行時の状態へ対応付けます。

Graphのノード、エッジ、状態バインディング、エントリーポイントはそれぞれ値として保持されます。

## 構造化領域

`StructuredRegion`は、Sequence、Repeat、Loop、Branchの入れ子構造と、その領域が読む状態効果を保持します。Loopは通常のグラフサイクルとして表さず、条件と領域効果をCompilerが検証できる形で保持します。構造化領域を下位変換した後も、ExecutablePlanへ渡す実行木と状態効果の対応を保ちます。

## 生成時点と利用時点

グラフはパイプラインやグラフビルダーがComponentを配置するときに生成されます。Compilerは契約スナップショットを読み、エントリーポイント、エッジ端点、ポート互換性、状態バインディング、構造化領域を検証します。コンパイラ規則がグラフを解決段階で変更する場合は、変更箇所を主張として宣言し、未申告の変更や競合を診断対象にします。

## 不変条件と診断

Graphのノード、エッジ、状態バインディング、エントリーポイントはそれぞれ値として保持され、未知のノードを参照する端点などは`invalid_graph_edge`や`invalid_entry_point`診断になります。 解決規則の未申告の変更や競合は`unclaimed_rewrite`または`conflicting_rewrite`診断になります。

## 拡張点

```python
from saealib.core import ComponentContract, ComponentGraph
from saealib.core import ComponentNode


class Empty:
    def contract(self) -> ComponentContract:
        return ComponentContract()


node = ComponentNode(component_id="root", component=Empty())
graph = ComponentGraph(nodes=(node,), entry_points=("root",))
```

この例は一つのノードを入口にしたGraphを作ります。 実際のポートとエッジの構築は、公開APIの構成に合わせて[APIリファレンス](../api/index.md)で確認してください。 Graph接続のルールを追加する場合は[フレームワーク拡張](extensions.md)を参照してください。 端点とentry pointの不変条件を保ち、Compilerの検証を迂回しないようにします。

## 関連ページ

[Contract](contract.md)は契約の包含を、[Compiler](compiler.md)はGraphの解決と検証を説明します。

## 参照

- {py:class}`saealib.core.ComponentGraph`
- {py:class}`saealib.core.Component`

