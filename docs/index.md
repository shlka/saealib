---
primary_layer: cross
related_layers: []
page_type: entry
---

# saealib

1回の評価に時間や費用がかかる目的関数のための、Pythonの最適化ライブラリです。
過去の評価結果から目的関数を近似するモデルを併用し、直接評価する回数を抑えて解を探索します。

```{button-ref} getting_started/quickstart
:ref-type: doc
:color: primary
:shadow:
:class: sd-mr-2

Quickstart →
```
```{button-ref} getting_started/what_is_saealib
:ref-type: doc
:color: secondary
:outline:

saealibとは
```

## ドキュメント

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`flag;sd-mr-1` はじめに
:link: getting_started/index
:link-type: doc
共通の使い始め方を確認します。
:::
:::{grid-item-card} {fa}`book-open;sd-mr-1` チュートリアル
:link: tutorials/index
:link-type: doc
用途別の手順を確認します。
:::
:::{grid-item-card} {fa}`cubes;sd-mr-1` 最適化の構成要素
:link: concepts/index
:link-type: doc
最適化を構成する要素を確認します。
:::
:::{grid-item-card} {fa}`sitemap;sd-mr-1` フレームワーク
:link: framework/index
:link-type: doc
契約、ComponentGraph、Compiler、Runtimeを確認します。
:::
:::{grid-item-card} {fa}`diagram-project;sd-mr-1` アルゴリズム
:link: algorithms/index
:link-type: doc
アルゴリズムの構成と出典を確認します。
:::
:::{grid-item-card} {fa}`code;sd-mr-1` APIリファレンス
:link: api/index
:link-type: doc
公開APIを参照します。
:::
::::

## 最小例

```python
from saealib import minimize
from saealib.benchmarks import rastrigin

result = minimize(rastrigin(n_var=10), max_fe=300, seed=0)
print(result.x, result.f)
```

どのコンポーネントを使うか、どのように組み合わせるかを設定できます。
構成の矛盾は、最適化を始める前に検証されます。
詳しくは[saealibとは](getting_started/what_is_saealib.md)と[最適化の構成要素](concepts/index.md)を参照してください。

インストール手順は[インストール](getting_started/installation.md)、実装の出典は[参考文献](references.md)にあります。

```{toctree}
:hidden:

getting_started/index
tutorials/index
concepts/index
framework/index
algorithms/index
references
api/index
```
