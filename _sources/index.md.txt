# saealib

**サロゲート支援進化的アルゴリズム (SAEA) の Python ライブラリ**

高コストな目的関数評価を軽量なサロゲートモデルで代替し，評価回数を大幅に削減します．

```{button-ref} getting_started/quickstart
:ref-type: doc
:color: primary
:shadow:
:class: sd-mr-2

クイックスタート →
```
```{button-ref} getting_started/what_is_saealib
:ref-type: doc
:color: secondary
:outline:

saealib とは？
```

---

::::{grid} 1 2 2 3
:gutter: 3
:margin: 4 4 0 0

:::{grid-item-card} シンプルな高レベル API
:link: getting_started/quickstart
:link-type: doc

```python
from saealib import minimize

result = minimize(func, dim=5,
                  lb=-5, ub=5)
```

`minimize()` / `maximize()` を呼ぶだけで動作．
:::

:::{grid-item-card} 柔軟な低レベル API
:link: user_guide/architecture
:link-type: doc

アルゴリズム・サロゲート・モデル管理戦略はすべてモジュール式で，自由に組み合わせられます．
:::

:::{grid-item-card} 拡張しやすい設計
:link: user_guide/custom_components
:link-type: doc

独自コンポーネントを実装してパイプラインに組み込めます．研究・実験用途に最適です．
:::

::::

---

## ドキュメント

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Getting Started
:link: getting_started/index
:link-type: doc

インストール・基本的な使い方・saealib の概念説明
:::

:::{grid-item-card} User Guide
:link: user_guide/index
:link-type: doc

アーキテクチャ詳細・コンポーネント解説・カスタマイズ方法
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

単目的・多目的最適化の実践チュートリアル
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

全クラス・関数のリファレンス
:::

::::


```{toctree}
:hidden:
:maxdepth: 1

getting_started/index
user_guide/index
tutorials/index
examples/index
api/index
```
