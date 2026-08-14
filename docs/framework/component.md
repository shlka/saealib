---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# コンポーネント（Component）

最適化の処理を実行するコンポーネントは、実行内容と静的な契約を分けて扱います。
`Component` Protocolは、そのコンポーネントが`ComponentContract`を返す境界だけを定めます。

## Component Protocolと`contract()`

`Component`は`contract() -> ComponentContract`を提供します。
`contract()`はコンパイル時に読み取る純粋な契約のスナップショットを返し、実行中の可変状態を返す場所ではありません。

`Component` Protocol自体は`execute()`を定義しません。
Graph-nativeの実行コンポーネントが`execute(StateView)`を提供することは、Runtimeが利用する別の実行境界です。

ComponentContractはComponentの基底クラスでも派生型でもありません。
Componentがコンポーネントを保持する場合、各コンポーネントの契約を`PartSpec`として親契約に包含します。

```python
from saealib.core import Component, ComponentContract


class Normalize(Component):
    def contract(self) -> ComponentContract:
        return ComponentContract()
```

この例はProtocolの最小境界だけを示します。
実際の入出力、状態、サービスは[ComponentContract](contract.md)と[宣言要素（Specs）](specs.md)で宣言します。

## PartSpecによる包含

`PartSpec`は、Componentがコンストラクターなどで保持する別のComponentの契約を名前付きで宣言します。
親Componentは子Componentを継承するのではなく、`parts`に子の契約を含めます。
`optional=True`は、そのコンポーネントが構成上省略可能であることを表します。

この包含により、Compilerは親のポートや状態効果と、保持コンポーネントの契約を同じ計画上で検証できます。
実行時のコンポーネントインスタンスの所有者はComponentであり、契約の不変性は`ComponentContract`が担保します。

## 生成時点と利用時点

ComponentのインスタンスはGraph構築時に`ComponentNode`へ配置されます。
Compilerはコンパイル開始時に`contract()`を読み取り、同じコンパイルの解決と検証ではそのスナップショットを使います。
Runtimeは契約を再解釈せず、ExecutablePlanが指定したポート、サービス、状態境界を使います。

## 責務外と拡張点

Componentは任意のStateStoreを直接変更せず、宣言したStateKeyのStateViewを読み、StatePatchまたはNodeResultを返します。
Componentはポートの互換性を自分で決めず、接続後の判定をCompilerへ委譲します。
新しいComponentを作る最小の拡張は、`contract()`とRuntimeが利用する実行境界を実装し、必要な契約を返すことです。

## 失敗と関連ページ

`contract()`がComponentContractを返さない場合、Graph構築またはCompilerの契約スナップショット取得で診断になります。
未宣言の状態アクセスやコンポーネント間の不整合は、[Contract](contract.md)、[Graph](graph.md)、[Compiler](compiler.md)の検証対象です。
拡張手順は[フレームワーク拡張](extensions.md)を、公開import経路は[APIリファレンス](../api/index.md)で確認してください。
