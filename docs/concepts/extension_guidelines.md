---
primary_layer: cross
related_layers: [layer3, layer4]
---

# 拡張ガイドライン

このページは、既存コンポーネントの軽量な差し替えと互換性用Pipelineの構成を扱います。
拡張は、既存コンポーネントの差し替えとフレームワークの拡張に分かれます。

- 既存の契約にコンポーネントを追加する場合は [独自Componentを実装する](../tutorials/custom_components.md) を参照してください。
- 契約、探索空間、Graph、Runtimeを追加する場合は [フレームワーク拡張](../framework/extensions.md) を参照してください。

import経路の規則は [Canonical Imports](../api/imports.md) にまとめています。

## 拡張点を選ぶ

変更したい責務に対応する最も浅い境界を選びます。
新しい実行契約が必要な場合だけFrameworkへ進み、既存の契約で表現できる差し替えはチュートリアルの手順で実装します。

| 変更したいこと | 主な境界 | 概念 | APIリファレンス |
|---|---|---|---|
| 候補生成や個体更新を変更する | `Algorithm` | [Algorithm](search_algorithms/algorithm.md) | [アルゴリズム](../api/algorithms.md) |
| 交叉、突然変異、選択を変更する | Operator | [Crossover](search_algorithms/crossover.md)、[Mutation](search_algorithms/mutation.md) | [演算子](../api/operators.md) |
| 予測モデルや学習データを変更する | `Surrogate`、`SurrogateManager` | [Surrogate](surrogate_modeling/surrogate.md) | [代理モデル](../api/surrogate.md) |
| 候補の選択や評価順序を変更する | `OptimizationStrategy`、`Stage` | [Strategy](execution_and_evaluation/strategies.md) | [戦略](../api/strategies.md) |
| 外部ライブラリを接続する | 明示的なAdapter | [Evaluation](execution_and_evaluation/evaluation.md) | [評価](../api/evaluation.md) |
| 候補表現や変数制約を追加する | `SearchSpace` | [SearchSpace](problem_and_ranking/search_space.md) | [Space](../api/space.md) |
| 入出力、状態、ライフサイクルを宣言する | `ComponentContract` | [Contract](../framework/contract.md) | [Coreリファレンス](../api/core.md) |
| 接続や互換性検査を変更する | `ComponentGraph`、Compiler rule | [Graph](../framework/graph.md)、[Compiler](../framework/compiler.md) | [Coreリファレンス](../api/core.md) |
| 実行、再開、非同期待機の意味を変更する | `ExecutionRuntime` | [Runtime](../framework/runtime.md) | [Executionリファレンス](../api/execution.md) |

フレームワーク契約は `saealib.core` から取得し、実装モジュールを安定した拡張APIとして扱いません。

`Algorithm`、`OptimizationStrategy`、`Surrogate`、`AcquisitionFunction`、`SurrogateManager`には、それぞれ差し替え可能な契約があります。
独自の探索や予測を実装するときは、対象のドメインコンポーネントを実装して `Optimizer.set_*()` から登録します。
既存コンポーネントの一部だけを変更する場合は、次の軽量な拡張手段を使えます。

## with_post / with_post_fit

[Crossover](search_algorithms/crossover.md)/[Mutation](search_algorithms/mutation.md)'s `with_post(fn)` and [Surrogate](surrogate_modeling/surrogate.md)'s `with_post_fit(fn)` add post-processing to an existing instance without subclassing.
They don't modify the original instance — they return a copy with the hook added.

Typical uses are adding a repair function to `Crossover`/`Mutation`, or post-fit processing on `Surrogate`.
Each component page's "Extension hooks" section has concrete examples per component.

## PipelineとStage

`Stage` と `Pipeline` には、互換性用の実行経路と構造化DSLの二つの境界があります。
Stageは `OptimizationState` を受け取り、更新した状態を返す互換性用の実行単位です。
一方、構造化Pipelineはgraph-native component、入れ子のPipeline、`Repeat`、`Loop`、`Branch`を保持し、状態を直接実行しません。

構造化PipelineへStageを入れるときは、`stage_component(stage)` で互換性境界を明示します。
Optimizerは内部でこのDSLをGraphへlowerし、構造化Runtimeで実行できる計画へ変換します。
通常の利用者がCompilerを直接呼び出す必要はありません。

`pipeline.replace("name", entry)` は構造上のエントリを置き換え、`pipeline.find("name", recursive=False)` は名前でエントリを検索します。

```python
from saealib import Pipeline, Stage
from saealib.stages import stage_component
from saealib.stages import CountGenerationStage


class DoubleCountStage(Stage):
    name = "count_generation"

    def execute(self, state):
        return state.replace(gen=state.gen + 2)


pipeline = Pipeline(
    steps=[
        stage_component(CountGenerationStage()),
        # graph-native componentや、別のStage adapterを続けて置く。
    ]
)
pipeline.replace("count_generation", stage_component(DoubleCountStage()))
```

[Stage](observation_and_state/stage.md) はStageの契約、組み込みStage、互換性用のカスタムStageを説明します。
このページでは、既存の構成を置き換えるときにどの境界を選ぶかを説明します。

## CallbackManager

[CallbackManager](observation_and_state/callbacks.md) is an observation mechanism that calls handlers when events fire.
Use `cbmanager.register/unregister/replace` to change the default pipeline's handlers at runtime.

The `candidates` field carried by `PostCrossoverEvent`/`PostMutationEvent`/`PostAskEvent` is for observation only; reassigning it inside a handler has no effect on the actual candidate array.
The distinction is: use `with_post` if you want to actually swap out the candidate array, and CallbackManager if you only need observation, logging, or conditional branching decisions.
See [CallbackManager](observation_and_state/callbacks.md)'s "The candidates field is for observation only" section for details.

The `Optimizer` instance itself isn't passed as a handler argument.
To swap a component at runtime, capture the `Optimizer` instance directly in the handler's closure and reassign `optimizer.algorithm`/`strategy`/`surrogate_manager`/`termination`.
Because each Strategy rebuilds the pipeline on every call to `step()`, this kind of swap reliably takes effect from the next generation (or the next iteration of `iterate()`) onward.

## Registry

`saealib.registry` is a mechanism for constructing an actual instance from a name (string) or a spec (`{"type": "Name", "params": {...}}`).
Where `with_post`/`Pipeline`-`Stage`/`CallbackManager` are mechanisms for "changing behavior at runtime," Registry serves a different purpose: "assembling components from strings or a config file."
Use it in situations that don't directly import classes, such as config-driven construction via preset YAML (`Optimizer.set_preset()`) or checkpoint resumption.

**`register(name=None)`** (decorator): Registers a class or function with the registry.
Adding `@register()` to a custom `Algorithm`/`Surrogate` subclass, etc., lets it be referenced by a short name just like a built-in component.

```python
from saealib import register
from saealib.surrogate.base import Surrogate


@register()
class MyCustomSurrogate(Surrogate):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    ...
```

`get`/`build`/`to_spec` are not exposed from saealib's top level; import them directly from `saealib.registry`.

**`get(name)`**: Resolves a registered name, or, if unregistered, a dotted path in `"module.submodule.ClassName"` form.

**`build(spec)`**: Recursively builds a spec into an actual instance.
If a value inside the spec is itself a nested spec, it is built recursively.
The form `{"callable": "dotted.path"}` resolves a function or built-in function itself (without calling it).

**`to_spec(obj)`**: The inverse of `build()`.
It reflects the constructor signature, reads same-named attributes, and recursively serializes them into a spec.
Classes with a `_registry_spec` attribute (such as `TerminationCondition`) don't use this generic reflection — they return that attribute directly.
This is the path `Optimizer.save_preset()` uses.

```python
from saealib.registry import build, get, to_spec

obj = build({"type": "MyCustomSurrogate", "params": {"alpha": 2.0}})
get("MyCustomSurrogate")  # -> the MyCustomSurrogate class
to_spec(obj)  # -> {"type": "MyCustomSurrogate", "params": {"alpha": 2.0}}
```

Several component pages have a note saying "class X is not `@register()`ed" — this only matters if you resolve classes by name via the Registry.

## どの仕組みを使うか

| What you want to do | Mechanism to use |
|---|---|
| Just add post-processing to an existing operator or surrogate | `with_post` / `with_post_fit` |
| Change the order of stages itself | `Pipeline` / `Stage` |
| External observation, logging, conditional swapping | `CallbackManager` |
| Assemble from a config file or preset | `Registry` |

## 関連コンポーネント

- [Crossover](search_algorithms/crossover.md) / [Mutation](search_algorithms/mutation.md) / [Surrogate](surrogate_modeling/surrogate.md): Components with `with_post`-style hooks
- [Stage](observation_and_state/stage.md): The contract of each stage that `Pipeline` combines
- [CallbackManager](observation_and_state/callbacks.md): The event list and observation mechanism
- [strategies](execution_and_evaluation/strategies.md): When the pipeline is rebuilt

## 参照

- {py:func}`saealib.register`
