# AccuracyBasedSurrogateSwitcher

サロゲートの精度に応じて`SurrogateManager`や`OptimizationStrategy`を動的に切り替える仕組みは、3つの独立した関心事の組み合わせでできています。

- **精度指標**（`SurrogateAccuracyMetric`）：1回のfit/predictペアからスカラー値を計算する
- **精度評価方法**（`AccuracyEvaluator`）：精度指標を使って、サロゲートの汎化性能をどう測るか（交差検証、held-outなど）を決める
- **切り替え判断**（`AccuracyBasedSurrogateSwitcher`）：評価結果を受けて、次に使うコンポーネントやパラメータを決める

精度評価方法は`GlobalSurrogateManager`などの`accuracy_evaluator`引数として注入され、その結果は`surrogate_manager.last_accuracy`プロパティを経由して切り替え判断へ渡されます。
この一連の仕組みは`iterate()`ループでの利用を前提としています。
高レベルAPI（`minimize`/`maximize`）は実行中のコンポーネント差し替え手段を提供しないため、これらのSwitcherは使えません。

## 精度指標: SurrogateAccuracyMetric

`SurrogateAccuracyMetric`が実装を要求するのは`name`（プロパティ）と`compute(y_true, y_pred) -> float`の2つです。

| クラス | 範囲 | 意味 |
|---|---|---|
| `SpearmanCorrelation` | `[-1, 1]`、高いほど良い | 目的ごとのSpearman順位相関の平均。EAではランクの保存性が重要という観点{cite}`yu2019spearman`に基づく |
| `RMSE` | `[0, ∞)`、低いほど良い | 二乗平均平方根誤差 |
| `R2Score` | `(-∞, 1]`、高いほど良い | 決定係数 |

`AccuracyEvaluator`のコンストラクタで`metrics`を省略すると、この3指標全てが既定として使われます。

## 精度評価方法: AccuracyEvaluator

`AccuracyEvaluator`が実装を要求するのは`evaluate(surrogate, train_x, train_y) -> SurrogateAccuracy`の1つだけです。

| クラス | パラメータ | 評価方法 |
|---|---|---|
| `KFoldAccuracyEvaluator` | `metrics=None, n_splits=5` | サロゲートをfoldごとに複製して再学習するk分割交差検証 |
| `LOOAccuracyEvaluator` | `metrics=None` | `n_splits=n_samples`の`KFoldAccuracyEvaluator`と等価（1件抜き交差検証） |
| `HeldOutAccuracyEvaluator` | `held_x, held_y, metrics=None` | 既にfit済みのサロゲートを再学習せず、指定したheld-outデータで評価する |

`HeldOutAccuracyEvaluator`は、直近の真の評価点との比較用途{cite}`hanawa2025switching`で使います。

`SurrogateAccuracy`（`metrics: dict[str, float]`, `n_samples: int`）は、評価結果を保持する単純なコンテナで、`get(name, default=nan)`でメトリクス名から値を取り出します。

## 切り替え判断: AccuracyBasedSurrogateSwitcher

`AccuracyBasedSurrogateSwitcher`が実装を要求するのは`switch(accuracy: SurrogateAccuracy | None) -> T`の1つだけです。
`iterate()`ループの中で`optimizer.set_*()`と組み合わせて呼びます。

```python
switcher = ManagerSwitcher(primary, fallback)
for ctx in optimizer.iterate():
    optimizer.set_surrogate_manager(
        switcher.switch(optimizer.surrogate_manager.last_accuracy)
    )
```

| クラス | パラメータ | 切り替え対象 |
|---|---|---|
| `ManagerSwitcher` | `primary, fallback, *, metric="spearman", threshold=0.5` | `SurrogateManager` |
| `StrategySwitcher` | `primary, fallback, *, metric="spearman", threshold=0.56` | `OptimizationStrategy` |
| `GenCtrlSwitcher` | `*, gm_max=5, gm_min=0, update_rate=0.5, metric="spearman", initial_error=0.5` | `gen_ctrl`（整数） |

`ManagerSwitcher`/`StrategySwitcher`は、指定したメトリクスが閾値以上なら`primary`、そうでなければ`fallback`を返す単純な二値切り替えです。
`StrategySwitcher`の既定閾値`0.56`は、{cite}`hanawa2025switching`によるPS/IB-GB切り替えの実装です。

`GenCtrlSwitcher`は、二値切り替えではなく`gen_ctrl`を指数平滑化で連続的に調整します{cite}`repicky2017genctrl`。
公開属性`smoothed_error`に平滑化済みの誤差推定値を保持する状態を持つため、1回の`run`につき1つのインスタンスを使います。
`GenCtrlSwitcher.switch()`が返す`int`は、そのまま[GenerationBasedStrategy](strategies.md)の`gen_ctrl`引数に渡す想定です。

## 独自Switcherの実装方法

独自の切り替えロジックが必要な場合は、`AccuracyBasedSurrogateSwitcher`を継承して`switch()`だけを実装すればよいです。

```python
from saealib import AccuracyBasedSurrogateSwitcher


class ThresholdIntSwitcher(AccuracyBasedSurrogateSwitcher):
    """spearman相関が0.7以上ならn_neighborsを増やす、という独自の整数パラメータ切り替え。"""

    def switch(self, accuracy):
        if accuracy is None:
            return 20
        return 50 if accuracy.get("spearman") >= 0.7 else 20
```

## 関連コンポーネント

- [SurrogateManager](surrogate_manager.md) — `accuracy_evaluator`引数と`last_accuracy`プロパティ
- [strategies](strategies.md) — `StrategySwitcher`/`GenCtrlSwitcher`が切り替える対象

## 参照

- {py:class}`saealib.AccuracyBasedSurrogateSwitcher`
- {py:class}`saealib.ManagerSwitcher`
- {py:class}`saealib.StrategySwitcher`
- {py:class}`saealib.GenCtrlSwitcher`
- {py:class}`saealib.SurrogateAccuracyMetric`
- {py:class}`saealib.SpearmanCorrelation`
- {py:class}`saealib.RMSE`
- {py:class}`saealib.R2Score`
- {py:class}`saealib.AccuracyEvaluator`
- {py:class}`saealib.KFoldAccuracyEvaluator`
- {py:class}`saealib.LOOAccuracyEvaluator`
- {py:class}`saealib.HeldOutAccuracyEvaluator`
- {py:class}`saealib.SurrogateAccuracy`
