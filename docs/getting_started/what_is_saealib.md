# saealibとは

saealibはサロゲート型進化的アルゴリズム(SAEA)のpython向け汎用ライブラリです。
進化的アルゴリズム(EA)やサロゲートモデル、モデル管理戦略などがモジュール化されており、組み合わせることでアルゴリズムを構築、実行します。

## SAEAとは

進化的アルゴリズム(EA)は生物の進化を模倣した最適化アルゴリズムです。
しかし進化の過程で、個体の評価を繰り返し行う必要があるため、高コスト最適化問題に対して課題があります。

SAEAはこの課題のために、軽量な数理モデルによる評価の代替を行い、高コストな評価の回数を削減します。

### 一般的なEA (GA)

<details>
<summary><b>一般的なEAの処理フロー（クリックで展開）</b></summary>

```{mermaid}
flowchart TD
    A[初期集団の生成] --> B["目的関数による評価"]
    B --> C[親の選択]
    C --> D[選択・交叉・突然変異]
    D --> E["目的関数による評価"]
    E --> F[次世代の選択]
    F --> G{終了条件?}
    G -- No --> C
    G -- Yes --> H[最良解を返す]

    style B fill:#e57373,color:#fff,stroke:#c62828
    style D fill:#e57373,color:#fff,stroke:#c62828
```

</details>

世代ごとに集団全体を評価するため、評価コストが大きくなります。

### SAEA （個体ベースGA）

<details>
<summary><b>SAEAの処理フロー（クリックで展開）</b></summary>

```{mermaid}
flowchart TD
    A[初期集団の生成] --> B["目的関数による評価"]
    B --> D[親の選択]
    D --> E[選択・交叉・突然変異]
    E --> F[サロゲートモデルの構築]
    F --> G["獲得関数によるスコアリング"]
    G --> H[有望な候補解の選択]
    H --> I["目的関数による評価"]
    I --> J{終了条件?}
    J -- No --> D
    J -- Yes --> K[最良解を返す]

    style B fill:#e57373,color:#fff,stroke:#c62828
    style H fill:#e57373,color:#fff,stroke:#c62828
    style F fill:#81c784,color:#fff,stroke:#2e7d32
```

</details>

真の評価を行う個体数を絞ることで、全体の評価コストを大幅に削減します。

## なぜsaealibが存在するのか

PythonでEAを使う場合、多目的探索の標準的な選択肢は**pymoo**です。
高コストな評価に特化したサロゲート併用最適化は、pymooの姉妹プロジェクトである**pysamoo**が担ってきましたが、開発は止まっています。
どちらのライブラリも、「どの候補解に高コストな真の評価を割り当てるか」という判断を、差し替え可能な部品としては扱っていません。
pymooは生成した候補解を常に真の関数で評価し、pysamooはこの判断をアルゴリズムクラスごとに直接書き込んでいます。

saealibは、この判断を**OptimizationStrategy**という第一級の差し替え可能なコンポーネントに切り出します。
individual-based/generation-based/pre-selection/directという4種の組み込み戦略はいずれもこの抽象を実装したものであり、独自の判断基準が必要であれば`OptimizationStrategy`を継承するだけで差し替えられます。
これに加えて、フィットと予測だけを行う**Surrogate**と、予測値をスコアへ変換する**AcquisitionFunction**を分離し、両者を**SurrogateManager**が仲介する構成を取っています。
サロゲートの実装を変えても獲得関数側のコードには影響せず、その逆も同様です。

| 比較項目 | saealib | pymoo | pysamoo |
|---|---|---|---|
| モデル管理戦略を差し替え可能な部品にしている | Yes | No（常に全候補を評価） | アルゴリズムクラスごとにハードコード |
| 型付きイベントによる実行中のコンポーネント差し替え | Yes | 「アルゴリズムをカスタマイズする用途ではない」（[公式ドキュメント](https://pymoo.org/interface/callback.html)） | No |
| サロゲートと獲得関数の分離 | Yes | No（pysamooに委譲） | 部分的 |
| ライセンス | Apache-2.0 | Apache-2.0 | AGPL-3.0 |

この差し替えを支えているのは、`Algorithm`/`OptimizationStrategy`/`Surrogate`/`AcquisitionFunction`/`SurrogateManager`がいずれも抽象基底を持ち、`Optimizer.set_*()`で構築時に差し替えられるという設計です。
サブクラス化するほどではない、軽い変更をしたい場面のためには、`with_post`/`with_post_fit`によるフックの追加、`Pipeline`/`Stage`によるステージの並べ替え、`CallbackManager`によるパイプラインの観察と実行時の差し替えという3つの軽量な機構も用意されています（詳細は[拡張のガイドライン](../components/extension_guidelines.md)を参照してください）。

候補解を生成してから母集団を更新するまでの流れは、**Ask-Tell**という形に分けられています。
`Algorithm.ask()`が候補解を生成し、`Algorithm.tell()`が母集団を更新するという2つのメソッドに分け、そのあいだのどの候補解が真の評価を受けるかという判断を`OptimizationStrategy.step()`に持たせています。
この分離によって、探索アルゴリズム本体を変えずに評価戦略だけを差し替える、という組み合わせが成立します。

こうして組み立てたパイプラインには、2つの入口が用意されています。
`minimize()`/`maximize()`は、このパイプラインをsensible defaultsで自動構成した、ゼロボイラープレートの高レベルAPIです。
`Optimizer`ビルダーと`.iterate()`ジェネレータによる低レベルAPIは、世代ごとの検査やカスタムループ制御が必要な研究用途に向きます。
両者は同じパイプラインの上に成り立っており、低レベルAPIだけが持つ独自機能はありません（組み立て方は[コンポーネント概要](../components/index.md)を参照してください）。

パイプライン全体でのスコアは、常に「高いほど良い」という規約に統一されています。
問題の最適化方向は`Problem`の`direction`（最小化なら-1、最大化なら+1）で表し、`Comparator`や結果抽出はこの符号を`weight`として使います。

現時点でのトレードオフも述べておきます。
saealibが提供する組み込みアルゴリズムはGAとPSOのみで、pymooやPlatEMOほどアルゴリズムの網羅性はありません。
これは設計の焦点をサロゲートと戦略層に絞ったことによるトレードオフです。
