# OptimizationState

最適化パイプラインを構成する各コンポーネントは、実行中の状態を直接保持せず、`OptimizationState`という1つの値オブジェクトを介してやり取りします。
`Algorithm.ask(ctx, ...)`や`Stage.execute(state)`など、ほぼ全てのシグネチャに`ctx`または`state`として登場します。

## OptimizationStateが表すもの

`OptimizationState`は、フィールドを直接書き換えるのではなく`replace()`で更新済みコピーを作って受け渡す、不変（immutable-style）な値オブジェクトとして設計されています。
`Initializer`が実行開始時に最初の状態を構築し、以後は各`Stage`が`replace()`で更新したコピーを次の`Stage`へ渡していきます。

ただし、この不変性には2つの制御された例外があります。

**`archive`**は追記専用（append-only）であり、in-placeの追記が許容されます。
評価のたびに複製すると評価回数の2乗に比例するコストになるため、意図的にこの部分だけ可変にしてあります。

**`rng`**は、呼び出すたびに内部状態が進むという副作用を持ちます。
これはNumPyの`Generator`自体の性質であり、`OptimizationState`固有の設計ではありません。

この2つ以外のフィールドは、`replace()`による不変的な更新が原則です。

## 主要フィールド

| フィールド | 内容 |
|---|---|
| `problem` | 解いている[Problem](problem.md) |
| `population` | 現世代の[Population](population.md) |
| `archive` | 評価済み解を蓄積する[Archive](population.md) |
| `pareto_archive` | 非優越解集合を維持する[ParetoArchive](population.md) |
| `rng` | 乱数生成器 |
| `fe` | 評価回数 |
| `gen` | 世代数 |
| `data` | ユーザー拡張用の自由形式の辞書 |

このほかに、パイプラインの各[Stage](stage.md)が読み書きする型付きフィールドがある。

| フィールド | 書き込むStage | 参照するStage |
|---|---|---|
| `offspring` | `AskStage` | 後続の各Stage |
| `evaluated_offspring` | `TrueEvaluationStage` | `ArchiveUpdateStage` |
| `scores` / `predictions` | `SurrogateScoreStage` | 後続の各Stage |

`data`はユーザー拡張用の辞書で、独自の`Stage`や`Callback`が任意の値を追加する場所として使います。
`state.data["key"] = value`という直接変更ではなく、`state.replace(data={**state.data, "key": value})`という形で新しい辞書を作って渡します。

## 便利プロパティ

`dim`/`n_obj`/`lb`/`ub`/`direction`/`comparator`は、いずれも`state.problem.xxx`への委譲プロパティです。
`ctx.problem.dim`と書く代わりに`ctx.dim`と書ける簡略表記として用意されています。

## replaceによる更新

`replace(**kwargs) -> OptimizationState`は`dataclasses.replace`のラッパーで、パイプライン中で最も頻繁に使われる更新方法です。
たとえば世代数を進める`CountGenerationStage`は、`state.replace(gen=state.gen + 1)`という形でフィールドを更新します。

`OptimizationState`には`fe`/`gen`をインクリメントする`count_fe(count=1)`/`count_generation()`という補助メソッドも用意されていますが、これらは`replace()`を経由しないその場限りのミューテーションです。
組み込みパイプラインの中でこの2つが実際に呼ばれるのは、[Initializer](initialization.md)が初期評価件数を`fe`へ加算する箇所だけであり、世代ごとの`gen`/`fe`の更新（`CountGenerationStage`/`TrueEvaluationStage`）はいずれも`replace()`を使う経路に統一されています。
独自の`Stage`を書く場合も、`replace()`を使う経路に揃えたほうが一貫性を保ちやすいです。

## チェックポイント

`save(path)`/`load(path, problem)`（後者はクラスメソッド）は、`OptimizationState`をnpz形式で保存し復元します。
保存対象は`archive`/`population`/`pareto_archive`の配列と`rng`の完全なbit-generator状態のみで、再現性は可能な限りの保証にとどまります（同一NumPyバージョン、同一環境内でのビット完全な再開を意図しているが、バージョンをまたぐ再現は保証しない）。

[NSGA3Comparator](comparators.md)が持つような、コンポーネント固有の内部rng（`state.rng`からspawnされたもの）は保存対象に含まれません。
再開時はそのような内部rngを`state.rng`から新しくspawnし直します。

自動的なチェックポイント保存の使い方は[チェックポイント](../tutorials/checkpoint.md)を参照してください。
ここで説明した`save`/`load`は、その内部で使われている`OptimizationState`自体の契約です。

## 関連コンポーネント

- [Population](population.md) — `population`/`archive`/`pareto_archive`フィールドの実体
- [Initializer](initialization.md) — 最初の`OptimizationState`を構築する
- [Stage](stage.md) — `state.replace(...)`で状態を更新しながらパイプラインを進める
- [チェックポイント](../tutorials/checkpoint.md) — `CheckpointCallback`経由の自動保存の使い方

## 参照

- {py:class}`saealib.OptimizationState`
