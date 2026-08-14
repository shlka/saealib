---
primary_layer: cross
related_layers: []
page_type: entry
---

# はじめに

saealibを使い始めるための準備と簡単な実行方法を紹介します。

## 基本情報

まずはここから始めましょう。
インストールと簡単な実装方法を紹介しています。

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

:::{grid-item-card} {fa}`rocket;sd-mr-1` Quickstart
:link: quickstart
:link-type: doc
最初の最適化を実行します。
:::

::::

## 利用方法を選ぶ

saealibは、どこまで自分で組み立てるかによって4つの使い方に分かれます。

| Layer | 利用方法 | 最初のステップ |
|---|---|---|
| Layer 1: Use | 既定の構成のまま最適化を実行。問題を定義し、`minimize()` / `maximize()` に渡します。 | [クイックスタート](quickstart.md)、[チュートリアル](../tutorials/index.md) |
| Layer 2: Compose | ビルトインのコンポーネントを選び、`Optimizer` で組み合わせます。文献上のアルゴリズムを再現します。 | [最適化の構成要素](../concepts/index.md)、[アルゴリズム](../algorithms/index.md) |
| Layer 3: Extend Components | 既存の責務に独自の実装を与えて差し替えます。抽象基底を継承します。 | [独自Componentを実装する](../tutorials/custom_components.md) |
| Layer 4: Extend Framework | 契約、グラフ、コンパイラ、実行の意味そのものを拡張します。 | [フレームワーク拡張](../framework/extensions.md) |

まずはLayer 1から始め、既定の構成で足りなくなった時点で下のLayerへ進みます。
変更したい要素からLayerを選ぶ場合は [拡張点の選び方](../concepts/extension_guidelines.md) を参照してください。

```{toctree}
:hidden:

what_is_saealib
installation
quickstart
```
