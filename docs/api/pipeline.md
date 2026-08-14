---
primary_layer: layer4
related_layers: [layer2, layer3]
page_type: reference
---

# Pipeline

`Pipeline` は、コンポーネントと構造化された制御領域を記述するDSLです。
`OptimizationState` を直接実行するオブジェクトではなく、Optimizerが実行計画を構築するための入力として使います。

```python
from saealib import Branch, Loop, Pipeline, Repeat

pipeline = Pipeline(
    name="generation",
    steps=[
        Repeat(
            Pipeline(steps=[ask, acquire, tell], name="surrogate_generation"),
            count=10,
            name="surrogate_generations",
        ),
        Loop(evaluation, until=budget_reached, name="evaluation_loop"),
        Branch(route, then=fast_path, else_=safe_path, name="route"),
    ],
)
```

この例はPipelineの構築だけを示しています。
通常の利用では、Strategyが作るGraphとPipelineを `Optimizer` がコンパイルし、利用者がCompilerを直接呼び出す必要はありません。

入れ子のPipelineと制御値は、名前付きの構造化領域として保持されます。
`Repeat` は固定回数の反復を表し、`Loop` と `Branch` は宣言された状態契約を通じて条件を評価します。
領域は通常のグラフサイクルへ単純化されないため、Runtimeは状態効果と再開フレームを保持できます。

構造化Pipelineへ置くComponentは、graph-nativeの `contract()` と `execute(StateView)` の境界を提供します。
旧来の `Stage` 実行面は互換性用Graph Builderへの接続点として残っています。
既存Stageを構造化Pipelineへ入れる場合は、`stage_component(stage)` で明示的に包みます。

コンパイル時には、必要な入力ポートを制御順序上流の互換性のある出力へ対応付けます。
一意に対応付けられた接続は `DataEdge` になり、接続がない場合や複数候補がある場合はCompiler診断になります。
曖昧な接続を解消する必要がある場合は、明示的なGraphを使います。

`stage_component(stage)` は移行用Adapterです。
トランザクションProxyを通じた状態書き込みは `StatePatch` へ変換されますが、ServiceやContext Capabilityとして公開された可変オブジェクトの所有権はAdapter側に残ります。
新しいgraph-native componentでは、永続的な可変状態を宣言済みのStateKeyとPatchの背後に置きます。

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   saealib.Stage
   saealib.Pipeline
   saealib.Repeat
   saealib.Loop
   saealib.Branch
```
