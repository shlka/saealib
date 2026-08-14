---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# 最適化状態（OptimizationState）

`OptimizationState` は、Optimizerの実行環境とチェックポイントを保持する値です。
Stage互換経路では、StageがこのStateを直接受け取り、`replace()` で更新したStateを次のStageへ渡します。

構造化ランタイムでは、graph-native componentが `OptimizationState` 全体を直接受け取ることはありません。
Componentは宣言したStateKeyだけを読む `StateView` を受け取り、変更を `StatePatch` として返します。
RuntimeはそのPatchをStateStoreへ適用し、次の実行境界を作ります。

## 責務と所有者

`OptimizationState`の責務は、Stage互換経路の実行値、利用者が確認する結果、再開に必要な値をまとめることです。
Component契約、Graph構造、Compiler ruleを保持する値ではありません。
状態の所有者はStage互換経路ではOptimizerとState、graph-native経路ではStateStoreです。

`OptimizationState`は`replace()`で新しい値を作り、`StateStore`はStatePatchを適用して新しい世代を作ります。
ComponentがStateStoreやOptimizationStateを直接変更する設計にすると、契約外の書き込みと再開位置の不整合をCompilerやRuntimeが追跡できません。

## 二つの状態境界

`OptimizationState` と `StateView` は、異なる実行境界を表します。

| 境界 | 受け取る値 | 返す値 | 主な利用者 |
|---|---|---|---|
| Stage互換境界 | `OptimizationState` | `OptimizationState` | 既存Stage、sequential compatibility runtime |
| graph-native境界 | 宣言済みの `StateView` | `StatePatch` または `NodeResult` | Component、Compiler、structured runtime |

この二つの境界を混同すると、graph-native componentから任意のStateを読み書きできるように見えてしまいます。
構造化Pipelineでは、ComponentContractが読むStateKeyと書くStateKeyを宣言し、Runtimeがその範囲だけをComponentへ渡します。

## OptimizationStateが保持する値

`OptimizationState` は、利用者が結果を確認したり、実行を再開したりするための値を保持します。

| 値 | 内容 |
|---|---|
| `problem` | 最適化するProblem |
| `population` | 現在のPopulationへの互換性用ショートカット |
| `archive` | 評価済み解を保持するArchiveへの互換性用ショートカット |
| `pareto_archive` | 非劣解集合への互換性用ショートカット |
| `rng` | 乱数生成器 |
| `fe` | 真の評価回数 |
| `gen` | 世代数 |
| `data` | Stage互換拡張用の追加データ |

`population`、`archive`、`pareto_archive` は、内部で名前付きのコレクションを参照します。
新しいgraph-native componentが任意の値を `data` へ追加する設計にはせず、StateContractでStateKeyを宣言します。

## StateStore、StateView、StatePatch

`StateStore` は、型付きの `StateKey` と値を対応付ける状態ストアです。
Componentへ渡す `StateView` は、ComponentContractが宣言した読み取りキーだけを公開します。

Componentが状態を変更するときは、直接Storeを変更せずに `StatePatch` を返します。
RuntimeはPatchを現在のStoreへ適用し、次の状態を生成します。
この方式によって、Componentが宣言していないStateを偶然に読み書きすることを防ぎます。

CompilerはStateContractのreads、writes、exportsとStateBindingの対応を確認します。
RuntimeはStateViewを宣言済みの読み取りキーに限定し、StatePatchの書き込みをStoreの型付きキーと世代へ適用します。
パッチが同じキーを競合させる場合や、存在しないキーを削除する場合の扱いはRuntimeのDiagnosticsに従います。

```python
from saealib.core import StatePatch, StateView
from saealib.core.state import RUNTIME_GENERATION


def execute(view: StateView) -> StatePatch:
    current = view.get(RUNTIME_GENERATION)
    return StatePatch(writes={RUNTIME_GENERATION: current + 1})
```

実際のStateKeyは文字列ではなく、`saealib.core.state` が提供する型付きキーを使います。
上のコードは、Viewから読み、Patchを返す境界だけを示す簡略化した例です。

## Stage互換経路での更新

カスタムStageは `OptimizationState` を受け取り、`state.replace(...)` で更新後のStateを返します。

```python
from saealib import Stage


class LogGenerationStage(Stage):
    name = "log_generation"

    def execute(self, state):
        print(state.gen)
        return state
```

Stage互換経路で追加の値を持たせる場合は、`state.data` をコピーしてから `replace()` に渡します。
graph-native componentで同じ値を扱う場合は、StateKeyとStatePatchを使います。

## 更新とチェックポイント

`replace(**kwargs) -> OptimizationState` は、Stage互換経路で値を更新するためのメソッドです。
`archive` は評価結果を蓄積するために変更され、`rng` は乱数を生成するたびに内部状態が進みます。
これらは、他の値を `replace()` で渡す設計と異なる、明示された例外です。

`save(path)` と `load(path, problem)` は、`OptimizationState` のnpzチェックポイントを扱います。
自動チェックポイントを使う場合は [Checkpointing](../../tutorials/checkpoint.md) を参照してください。
pickle形式ではOptimizer側の状態も必要になるため、保存経路は `CheckpointCallback` と `Optimizer` の組み合わせで決まります。

## どの状態境界を使うか

- 既存のStageを実装する：`OptimizationState` と `replace()` を使う。
- 新しいgraph-native Componentを実装する：`ComponentContract`、`StateView`、`StatePatch` を使う。
- Optimizerの実行結果を確認する：`Optimizer.run()` または `Optimizer.iterate()` が返す `OptimizationState` を使う。
- 実行を再開する：`saealib.context.OptimizationState.load()` と `Optimizer.run_from()` を使う。

## 代表的な失敗

Stage互換経路で`replace()`の対象を取り違えると、次のStageが期待する値を受け取れません。
graph-native経路でStateContractにないキーを読む、書く、または公開すると、Compilerの状態効果検証で診断になります。
チェックポイントを異なるProblemや互換しない計画へ適用すると、保存した状態と実行境界が一致しません。

## 関連ページ

- [Stage](stage.md)：`OptimizationState` を直接受け取る互換性用の実行単位
- [Framework](../../framework/index.md)：Component、Contract、Graph、Compilerの関係
- [Runtime](../../framework/runtime.md)：Plan、Runtime、StatePatchの適用境界
- [Population](population.md)：Population、Archive、ParetoArchiveのデータ構造
- [Checkpointing](../../tutorials/checkpoint.md)：チェックポイントの保存と再開

## 参照

- {py:class}`saealib.context.OptimizationState`
- {py:class}`saealib.core.StateStore`
- {py:class}`saealib.core.StateView`
- {py:class}`saealib.core.StatePatch`
- [Framework ComponentContract](../../framework/contract.md)：状態宣言を含む契約
- [Framework Compiler](../../framework/compiler.md)：状態効果の検証とDiagnostics
