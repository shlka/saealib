---
primary_layer: cross
---

# saealibとは

saealibは、サロゲートモデルを使う進化計算のためのPythonライブラリです。目的関数の評価コストが高い最適化問題を対象とします。

## SAEAとは

進化計算では、候補解を生成し、目的関数で評価し、その結果を使って次の候補解を作る処理を繰り返します。目的関数の評価に時間や費用がかかる場合、多数の候補解を直接評価することが大きな負担になります。

SAEAは、過去の評価から目的値を推定するサロゲートモデルを使い、直接評価に送る候補を絞り込みます。サロゲートの予測は低コストですが近似値なので、最終的な判断には目的関数の直接評価を使います。

```{mermaid}
flowchart TD
    A[候補を生成] --> B[サロゲートモデルで予測]
    B --> C[実評価する候補を選択]
    C --> D[目的関数で評価]
    D --> E[モデルと探索を更新]
    E --> A
```

## saealibの特徴

候補解の生成、代理モデル、評価対象の選択といった処理が分離されており、それぞれを交換できます。`OptimizationStrategy`は、どの候補を直接評価するかを決める独立した戦略です。

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

:::{grid-item-card} {fa}`cubes;sd-mr-1` 最適化コンポーネント
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
