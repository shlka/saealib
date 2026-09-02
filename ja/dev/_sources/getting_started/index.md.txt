---
primary_layer: cross
related_layers: []
page_type: entry
---

# はじめに

saealibを使い始めるための準備と簡単な実行方法を紹介します。

## 基本情報

まずはここから始めましょう。

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`circle-info;sd-mr-1` saealibとは
:link: what_is_saealib
:link-type: doc
saealibの目的と全体像を確認します。
:::

:::{grid-item-card} {fa}`download;sd-mr-1` インストール
:link: installation
:link-type: doc
saealibを環境に導入します。
:::

:::{grid-item-card} {fa}`rocket;sd-mr-1` クイックスタート
:link: quickstart
:link-type: doc
最初の最適化を実行します。
:::

::::

(choose-your-layer)=
## saealibの使い方を選ぶ

saealibには、自分で組み立てる構成の量に応じて4つの使い方があります。

| レイヤー | 使い方 | 最初のステップ |
|---|---|---|
| Layer 1: Use | 既定の構成で最適化を実行します。問題を定義し、`minimize()`または`maximize()`に渡します。 | [クイックスタート](quickstart.md)、[チュートリアル](../tutorials/index.md) |
| Layer 2: Compose | ビルトインコンポーネントを選び、`Optimizer`と組み合わせます。文献にあるアルゴリズムも再現できます。 | [Optimizationコンポーネント](../concepts/index.md)、[アルゴリズム](../algorithms/index.md) |
| Layer 3: Extend Components | 抽象基底クラスを継承し、既存の責務を独自実装に置き換えます。 | [独自Componentを実装する](../tutorials/custom_components.md) |
| Layer 4: Extend Framework | 契約、グラフ、コンパイラ、実行そのものの意味を拡張します。 | [フレームワーク拡張](../framework/extensions.md) |

Layer 1から始め、既定値で足りない場合に下位のLayerへ進みます。変更したい部分からLayerを選ぶ場合は、[拡張点の選び方](../concepts/extension_guidelines.md)を参照してください。

```{toctree}
:hidden:

what_is_saealib
installation
quickstart
```
