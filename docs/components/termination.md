# Termination

`Optimizer`は、最適化をいつ止めるかという判断を、`Termination`という差し替え可能なトップレベルコンポーネントに委ねています。
`Optimizer.set_termination(termination)`で渡し、実行時は世代末に`is_terminated(ctx) -> bool`が呼ばれます。

## Terminationの役割

`Termination(*conditions)`は1つ以上の条件を受け取り、そのいずれか1つでも真になった時点で終了します（OR結合）。
コンストラクタに複数条件を渡す書き方は、後述する`any_of`と同じ意味になります。

```python
from saealib import Termination, max_fe, max_gen

termination = Termination(max_fe(2000), max_gen(100))
```

クラスメソッドも用意されています。

| クラスメソッド | 意味 |
|---|---|
| `Termination.any_of(*conditions)` | いずれか1つで終了（`Termination(*conditions)`と同義） |
| `Termination.all_of(*conditions)` | 全て満たすまで終了しない |
| `Termination.not_(condition)` | 条件を満たさない間だけ終了しない |

## 組み込み終了条件

`Termination`に渡す各条件は、`TerminationCondition`という薄いラッパーです。
組み込みのファクトリ関数はいずれも`TerminationCondition`を返します。

| 関数 | 終了条件 |
|---|---|
| `max_fe(value)` | 評価回数が`value`に到達（`ctx.fe >= value`） |
| `max_gen(value)` | 世代数が`value`に到達（`ctx.gen >= value`） |
| `f_target(value)` | アーカイブの最良目的値が`value`に到達。単目的向けで、`ctx.direction`から最小化/最大化を自動判定する。アーカイブが空の間は終了しない |
| `stalled(window, tol=1e-8)` | `window`世代連続で改善が`tol`を超えなければ終了 |

`stalled`が返す`TerminationCondition`は、内部に「これまでの最良スコア」をクロージャで保持する状態付きの条件です。
1回の`run`につき1つのインスタンスを使う想定であり、複数の`run`で使い回すと前回の状態が残ってしまいます。

## Terminationの拡張方法

`TerminationCondition`は抽象基底クラスではなく、任意のcallableを受け取って`|`(OR)/`&`(AND)/`~`(NOT)という演算子オーバーロードを提供する合成用のラッパーです。
`OptimizationState -> bool`を返すプレーンな関数はどこでも自動的に`TerminationCondition`へ変換されるため、独自の終了条件を追加するのに基底クラスを継承する必要はありません。

```python
from saealib import Termination, max_fe


def my_condition(ctx):
    f = ctx.archive.get("f")
    return f is not None and len(f) > 0 and f.min() < 1e-6


termination = Termination(max_fe(2000), my_condition)
```

名前や説明文を明示的に持たせたい場合だけ、`TerminationCondition(func, name=..., doc=...)`で明示的にラップします。

複数の条件は演算子で宣言的に組み合わせられます。

```python
from saealib import max_fe, max_gen, stalled

both = max_fe(2000) & max_gen(100)  # 両方満たすまで続ける
either = max_gen(100) | max_fe(2000)  # 早い方で終了
not_stalled = ~stalled(20)  # stallしていない間
```

```{note}
`TerminationCondition`のコンストラクタが持つ`spec`引数は、[Registry](extension_guidelines.md)がプリセットへシリアライズする際に使う内部フィールドであり、通常のユーザーコードでは意識しなくてよい。
```

## 関連コンポーネント

- [OptimizationState](optimization_state.md) — 各終了条件が受け取る`ctx`
- [拡張のガイドライン](extension_guidelines.md) — `Registry`による設定駆動構築

## 参照

- {py:class}`saealib.Termination`
- {py:class}`saealib.TerminationCondition`
- {py:func}`saealib.max_fe`
- {py:func}`saealib.max_gen`
- {py:func}`saealib.f_target`
- {py:func}`saealib.stalled`
