# saealib

Pythonで実装された、**サロゲート型進化的アルゴリズム(SAEA)** のための総合的なライブラリです。

目的関数の評価コストが高い最適化問題を想定して設計されており、`saealib`は進化的アルゴリズム・サロゲートモデル・モデル管理戦略を組み合わせるモジュール式のフレームワークを提供します。

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

saealibとは？
```

---

::::{grid} 1 2 2 3
:gutter: 3
:margin: 4 4 0 0

:::{grid-item-card} {fa}`bolt;sd-mr-1` 高レベルAPI
:link: getting_started/quickstart
:link-type: doc

```python
from saealib import minimize

result = minimize(func, dim=5,
                  lb=-5, ub=5)
```

`minimize()` / `maximize()`による、定型コード不要の高レベルAPI。
:::

:::{grid-item-card} {fa}`sliders;sd-mr-1` 低レベルAPI
:link: components/index
:link-type: doc

`Optimizer`ビルダーと`iterate()`ジェネレータにより、研究用途向けに世代ごとの検査やカスタムループ制御が可能です。
:::

:::{grid-item-card} {fa}`puzzle-piece;sd-mr-1` 拡張性
:link: components/index
:link-type: doc

すべての概念は抽象基底クラスを持ち、構築時に差し替え可能です。
そのため、フォークすることなくあらゆるSAEAバリアントを表現できます。
:::

::::

---

## ドキュメント

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {fa}`flag;sd-mr-1` はじめに
:link: getting_started/index
:link-type: doc

saealibを初めて使う方へ。インストール方法、基本的な使い方、コアとなる概念を説明します。
:::

:::{grid-item-card} {fa}`book-open;sd-mr-1` チュートリアル
:link: tutorials/index
:link-type: doc

状況別のセットアップガイド: 単目的/多目的最適化、制約、チェックポイント。
:::

:::{grid-item-card} {fa}`cubes;sd-mr-1` コンポーネント
:link: components/index
:link-type: doc

各コンポーネントの詳しい使い方と拡張ガイドライン。
:::

:::{grid-item-card} {fa}`diagram-project;sd-mr-1` アルゴリズム
:link: algorithms/index
:link-type: doc

文献上の名前がついたアルゴリズムを、saealibのコンポーネントの組み合わせとしてどう再現するか。
:::

:::{grid-item-card} {fa}`bookmark;sd-mr-1` 文献リファレンス
:link: references
:link-type: doc

実装済みのアルゴリズム・演算子・比較方法の理論的出典をまとめた文献リスト。
:::

:::{grid-item-card} {fa}`code;sd-mr-1` API Reference
:link: api/index
:link-type: doc

すべてのクラス・関数の完全な仕様。
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

getting_started/index
tutorials/index
components/index
algorithms/index
references
api/index
```
