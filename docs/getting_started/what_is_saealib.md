---
primary_layer: cross
---

# saealibとは

saealibは、代理モデルを活用する進化計算を扱うPythonライブラリです。
目的関数の評価に大きな計算コストがかかる最適化問題を対象としています。

## SAEAとは

進化計算では、候補解を生成し、目的関数で評価し、その結果を使って次の候補解を作る処理を繰り返します。
目的関数の評価に時間や費用がかかる場合、候補解を何度も直接評価することが大きな負担になります。

SAEAは、過去の評価結果から目的関数の値を推定する代理モデルを使い、直接評価する候補解を絞り込みます。
代理モデルによる予測は安価ですが近似であり、最終的な判断には目的関数による直接評価を使います。

```{mermaid}
flowchart TD
    A[Generate candidates] --> B[Predict with surrogate model]
    B --> C[Select candidates for true evaluation]
    C --> D[Evaluate with objective function]
    D --> E[Update model and search]
    E --> A
```

## saealibの特徴

候補解の生成、代理モデル、評価対象の選択といった処理が分離されており、それぞれを交換できます。
`OptimizationStrategy`は、どの候補解を直接評価するかを決める独立した戦略です。

## 次に読む

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {fa}`rocket;sd-mr-1` クイックスタート
:link: ../getting_started/quickstart
:link-type: doc
最小構成で最初の最適化を実行します。
:::

:::{grid-item-card} {fa}`graduation-cap;sd-mr-1` チュートリアル
:link: ../tutorials/index
:link-type: doc
目的に合わせた使い方を例から学びます。
:::

:::{grid-item-card} {fa}`cubes;sd-mr-1` 最適化の構成要素
:link: ../concepts/index
:link-type: doc
saealibを構成するコンポーネントの役割を確認します。
:::

:::{grid-item-card} {fa}`diagram-project;sd-mr-1` アルゴリズム
:link: ../algorithms/index
:link-type: doc
利用できるアルゴリズムの構成を確認します。
:::

::::
