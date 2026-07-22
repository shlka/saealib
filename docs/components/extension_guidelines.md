# 拡張のガイドライン

`Algorithm`/`OptimizationStrategy`/`Surrogate`/`AcquisitionFunction`/`SurrogateManager`はいずれも抽象基底を持ち、`Optimizer.set_*()`で丸ごと差し替えられます。
これは正攻法であり、独自の探索アルゴリズムやサロゲートが必要な場合はこの経路を使います。
一方で、既存コンポーネントを丸ごと差し替えるほどではない変更をしたい場面向けに、4つの軽量な機構が用意されています。

## with_post / with_post_fit

[Crossover](crossover.md)/[Mutation](mutation.md)の`with_post(fn)`と、[Surrogate](surrogate.md)の`with_post_fit(fn)`は、サブクラス化せずに既存インスタンスへ後処理を追加します。
元のインスタンスは変更せず、フックを追加したコピーを返します。

用例は、`Crossover`/`Mutation`への修復関数の追加や、`Surrogate`のフィット後処理です。
各コンポーネントページの「拡張フック」節に、コンポーネントごとの具体例があります。

## Pipeline / Stage

`OptimizationStrategy`の内部世代ループは、[Stage](stage.md)という単位の列を`Pipeline`で順に実行する形で構成されています。
`pipeline.replace("name", stage)`は特定のステージを別のステージに差し替え、`pipeline.find("name", recursive=False)`は`name`でステージを検索します。

```python
from saealib import Pipeline, Stage
from saealib.stages import CountGenerationStage


class DoubleCountStage(Stage):
    name = "count_generation"

    def execute(self, state):
        return state.replace(gen=state.gen + 2)


pipeline = Pipeline([CountGenerationStage(), ...])
pipeline.replace("count_generation", DoubleCountStage())
```

各Stageが満たす契約、組み込み11種の一覧、独自Stageの実装方法は[Stage](stage.md)を参照してください。
このページでは「ステージの並べ替えと差し替え」という操作自体を扱います。

## CallbackManager

[CallbackManager](callbacks.md)は、イベント発火時にハンドラを呼ぶ観察用の仕組みです。
`cbmanager.register/unregister/replace`でデフォルトパイプラインのハンドラを実行時に変更します。

`PostCrossoverEvent`/`PostMutationEvent`/`PostAskEvent`が持つ`candidates`フィールドは観測目的であり、ハンドラ内で再代入しても実際の候補配列には反映されません。
候補配列そのものを差し替えたいなら`with_post`、観測やログ、条件付きの分岐判断だけならCallbackManager、という使い分けになります。
詳細は[CallbackManager](callbacks.md)の「candidatesフィールドは観測目的」節を参照してください。

`Optimizer`インスタンス自体はハンドラの引数として渡されません。
実行時にコンポーネントを差し替えたい場合は、ハンドラのクロージャで`Optimizer`インスタンスを直接捕捉し、`optimizer.algorithm`/`strategy`/`surrogate_manager`/`termination`を再代入します。
各Strategyは`step()`のたびにパイプラインを再構築するため、この差し替えは次世代（または`iterate()`の次イテレーション）から確実に反映されます。

## Registry

`saealib.registry`は、名前（文字列）またはspec（`{"type": "Name", "params": {...}}`）から実インスタンスを構築する仕組みです。
`with_post`/`Pipeline-Stage`/`CallbackManager`が「実行時の挙動を変える」ための機構であるのに対し、Registryは「コンポーネントを文字列や設定ファイルから組み立てる」ための機構であり、目的が異なります。
プリセットYAML経由の設定駆動構築（`Optimizer.set_preset()`）やチェックポイント再開のように、クラスを直接importしない場面で使います。

**`register(name=None)`**（デコレータ）：クラスや関数をレジストリへ登録します。
独自の`Algorithm`/`Surrogate`等のサブクラスに`@register()`を付けるだけで、組み込みコンポーネントと同じように短い名前で参照できるようになります。

```python
from saealib import register
from saealib.surrogate.base import Surrogate


@register()
class MyCustomSurrogate(Surrogate):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    ...
```

`get`/`build`/`to_spec`は`saealib`のトップレベルからは公開されておらず、`saealib.registry`から直接importします。

**`get(name)`**：登録名、または未登録なら`"module.submodule.ClassName"`形式のドットパスとして解決します。

**`build(spec)`**：specを再帰的に実インスタンスへ構築します。
spec内の値がさらに入れ子のspecであれば再帰的に構築されます。
`{"callable": "dotted.path"}`という形式は、関数や組み込み関数そのものを（呼び出さずに）解決します。

**`to_spec(obj)`**：`build()`の逆操作。
コンストラクタシグネチャを反映し、同名属性を読んでspecへ再帰的にシリアライズします。
`_registry_spec`属性を持つクラス（`TerminationCondition`など）はこの汎用リフレクションを使わず、その属性を直接返します。
`Optimizer.save_preset()`が使う経路です。

```python
from saealib.registry import build, get, to_spec

obj = build({"type": "MyCustomSurrogate", "params": {"alpha": 2.0}})
get("MyCustomSurrogate")  # -> MyCustomSurrogate クラス
to_spec(obj)  # -> {"type": "MyCustomSurrogate", "params": {"alpha": 2.0}}
```

各コンポーネントページで「◯◯クラスは`@register()`未登録」という注記が複数箇所にありますが、いずれもRegistry経由でクラス名から解決する使い方をする場合にのみ影響します。

## 使い分けの指針

| やりたいこと | 使う機構 |
|---|---|
| 既存の演算子やサロゲートに後処理を足すだけ | `with_post` / `with_post_fit` |
| ステージの並び自体を変えたい | `Pipeline` / `Stage` |
| 外からの観測、ログ、条件付きの差し替え | `CallbackManager` |
| 設定ファイルやプリセットから組み立てたい | `Registry` |

## 関連コンポーネント

- [Crossover](crossover.md) / [Mutation](mutation.md) / [Surrogate](surrogate.md) — `with_post`系フックを持つコンポーネント
- [Stage](stage.md) — `Pipeline`が組み合わせる各ステージの契約
- [CallbackManager](callbacks.md) — イベント一覧と観測の仕組み
- [strategies](strategies.md) — パイプラインの再構築タイミング

## 参照

- {py:func}`saealib.register`
