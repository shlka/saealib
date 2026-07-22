# Stage

組み込みの`OptimizationStrategy`（IB/GB/PS/Direct）は、1世代分の処理を`Stage`という単位に分割し、`Pipeline`で順に実行する形で構成されています。
[拡張のガイドライン](extension_guidelines.md)の「Pipeline/Stage」節では`pipeline.replace`/`find`によるステージの並べ替え方を扱いました。
このページでは、各`Stage`が満たす契約、組み込み11種の詳細、独自`Stage`の実装方法を扱います。

## Stageの役割

`Stage`が実装を要求するメソッドは`execute(state: OptimizationState) -> OptimizationState`の1つだけです。
`OptimizationState`を受け取り、1つの整った処理を行い、更新した（あるいは同じ）状態を返します。

クラス属性は3つあります。

- **`name`**：`pipeline["name"]`によるルックアップに使うマシン可読な識別子
- **`label`**：人間可読な説明
- **`notation`**：`to_pseudocode()`が使うLaTeX記法

## Pipeline

`Pipeline(Stage)`は、`Stage`のリストを`functools.reduce`で順次実行する合成クラスです。
`Pipeline`自体も`Stage`のサブクラスなので、他の`Pipeline`にネストできます。

| 操作 | 内容 |
|---|---|
| `pipeline["name"]` | `name`でステージを検索する |
| `pipeline.replace(name, stage)` | `name`のステージを別の`Stage`に差し替える |
| `pipeline.find(name, *, recursive=False)` | `recursive=True`で`SurrogateOnlyLoopStage`のような入れ子ステージの内部も探索する |

## 組み込みStage

`OptimizationState`の`offspring`/`scores`/`predictions`/`evaluated_offspring`という標準フィールドを、各Stageがどう読み書きするかを示す。

| クラス | 読む | 書く |
|---|---|---|
| `CountGenerationStage()` | — | `gen` |
| `AskStage(algorithm, n_offspring=None, cbmanager=None)` | — | `offspring` |
| `SurrogateScoreStage(surrogate_manager, cbmanager=None, *, refit=True)` | `offspring` | `scores`, `predictions` |
| `SurrogateFitStage(surrogate_manager)` | `archive` | — |
| `TopKSelectionStage(k)` | `offspring`, `scores` | `offspring`（上位k件のみ） |
| `SortByScoreStage()` | `offspring`, `scores` | `offspring`, `scores`（全件を降順に並べ替え） |
| `TrueEvaluationStage(evaluator, cbmanager=None, n_eval=None)` | `offspring` | `evaluated_offspring`, `fe` |
| `ArchiveUpdateStage()` | `evaluated_offspring` | `archive`, `pareto_archive` |
| `TellStage(algorithm)` | `offspring` | `population` |
| `SurrogateOnlyLoopStage(algorithm, surrogate_manager, gen_ctrl, cbmanager=None)` | — | 内部ループ全体 |
| `InitializationStage(initializer, provider, problem)` | — | 状態全体を再構築 |

`AskStage`は`algorithm.ask()`を呼んで`state.offspring`に書き込み、`cbmanager`経由でPostCrossover/PostMutation/PostAskEventを発火します。

`SurrogateScoreStage`は`surrogate_manager.score_candidates()`でスコアリングし、`state.scores`/`state.predictions`に書き込むと同時に、各候補の`tell_f`も設定します。

`SurrogateFitStage`は、アーカイブが変化しない内部ループの前に1回だけサロゲートを事前フィットするために使います。
下流の`SurrogateScoreStage`に`refit=False`を渡すのとセットで使います。

`TopKSelectionStage`は`state.scores`の降順で上位k件だけを`state.offspring`に残し、残りは破棄します。
`SortByScoreStage`は`TopKSelectionStage`と異なり全候補を保持したまま降順に並べ替えるだけで、IB系戦略が使います。

`TrueEvaluationStage`は`state.offspring`の先頭`n_eval`件（`None`なら全件、`int`または`Callable[[OptimizationState], int]`を指定できる）を真の目的関数で評価します。

`SurrogateOnlyLoopStage`は、`GenerationBasedStrategy`が使う複合ステージです。
`CountGeneration → Ask → SurrogateScore(refit=False) → Tell`という内部ループを`gen_ctrl`回繰り返します。
`gen_ctrl=0`のときはno-opになります。

```{note}
`InitializationStage`の`execute()`に渡された`state`引数は無視され、常に初期化から新しい状態を作ります。
ユーザー定義パイプラインの先頭で、初期化自体をパイプラインの一部として扱いたい場合に使います。
```

組み込み4種のStrategy（IB/GB/PS/Direct）がこれらのStageをどう組み合わせてパイプラインを構成するかは、[strategies](strategies.md)と[コンポーネント概要](index.md)のパイプライン図を参照してください。
このページでは各Stage単体の契約を扱います。

## 独自Stageの実装方法

独自のパイプラインステージが必要な場合は、`Stage`を継承して`execute()`だけを実装すればよいです。
`state.replace(...)`で不変的に新しい状態を作る、`OptimizationState`の更新パターンに従います。

```python
from saealib import Stage


class LogGenerationStage(Stage):
    """世代番号を標準出力に記録するだけのカスタムステージ。"""

    name = "log_generation"
    label = "Log generation number"

    def execute(self, state):
        print(f"generation {state.gen}")
        return state
```

独自のフィールドを持ち回したい場合は、`OptimizationState`の`data`という拡張用の辞書を使います。
`state.replace(data={**state.data, "key": value})`という形で値を追加します。

## `to_pseudocode`

`to_pseudocode(expand=False, indent=0)`は、各Stageの`notation`を論文用の擬似コード（LaTeX algorithmic記法）として出力する機構です。
`AskStage`/`TellStage`/`SurrogateOnlyLoopStage`は、`expand=True`のとき`Algorithm.ask_notation`/`tell_notation`を展開する独自実装を持ちます。

## 関連コンポーネント

- [拡張のガイドライン](extension_guidelines.md) — `pipeline.replace`/`find`によるステージの並べ替え方
- [strategies](strategies.md) — 組み込みStrategyがStageをどう組み合わせるか
- [OptimizationState](optimization_state.md) — `execute()`が読み書きする状態オブジェクト
- [コンポーネント概要](index.md) — パイプライン全体の構成図

## 参照

- {py:class}`saealib.Stage`
- {py:class}`saealib.Pipeline`
- {py:class}`saealib.CountGenerationStage`
- {py:class}`saealib.AskStage`
- {py:class}`saealib.SurrogateScoreStage`
- {py:class}`saealib.SurrogateFitStage`
- {py:class}`saealib.TopKSelectionStage`
- {py:class}`saealib.SortByScoreStage`
- {py:class}`saealib.TrueEvaluationStage`
- {py:class}`saealib.ArchiveUpdateStage`
- {py:class}`saealib.TellStage`
- {py:class}`saealib.SurrogateOnlyLoopStage`
- {py:class}`saealib.InitializationStage`
