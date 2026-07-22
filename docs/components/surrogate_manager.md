# SurrogateManager

[Surrogate](surrogate.md)がfit/predictだけを担うのに対し、`SurrogateManager`はfit、predict、スコアリングのパイプライン全体を協調させます。
`score_candidates()`はスカラースコアだけでなく元の予測値も返すため、呼び出し側（`IndividualBasedStrategy`など）は予測目的関数値を子個体へ割り当てられます。

`Optimizer.set_surrogate_manager()`は、`Optimizer.set_surrogate()`（[Surrogate](surrogate.md)を`LocalSurrogateManager`でラップする簡易版）とは別のトップレベル差し替え点です。

## SurrogateManagerの役割

`SurrogateManager`の抽象メソッドは`score_candidates()`の1つだけで、残りはフックとして既定実装を持ちます。

**`score_candidates(candidates_x, archive, ctx=None, *, refit=True) -> tuple[np.ndarray, list[SurrogatePrediction]]`**（抽象）：候補群をスコアリングします。
`refit=True`（既定）ではスコアリング前にサロゲートを学習し直します。

**`fit(archive, ctx=None) -> None`**：既定はno-op。
`score_candidates(..., refit=False)`を連続で呼ぶ前に1回だけ呼ぶプレフィット用のフックで、アーカイブが変化しない場面（`GenerationBasedStrategy`のサロゲートのみ内部ループなど）で使います。

**`last_accuracy: SurrogateAccuracy | None`**（クラス属性）：直近の`fit`が計算した精度指標。
詳細は[サロゲート精度評価と動的切り替え](surrogate_switching.md)で扱います。

**`iter_acquisitions() -> Iterator[AcquisitionFunction]`**：`Optimizer`が`problem.direction`を各[AcquisitionFunction](acquisition_functions.md)へ自動注入するために使う内部フック。
既定は`self.acquisition`があればそれを1つ返します。
`PairwiseSurrogateManager`のように獲得関数を持たないマネージャーは何も返さず、`CompositeSurrogateManager`はサブマネージャーへ委譲します。

**`post_score(scores, predictions, ctx=None)`** / **`with_post_score(fn)`**：スコアリング後処理のライフサイクルフック。
[Surrogate](surrogate.md)の`with_post_fit`と同型で、`with_post_score`は元のインスタンスを変更せずコピーにフックを追加します。

**`on_generation_end(gen, archive, ctx=None)`** / **`with_on_generation_end(fn)`**：世代末フック。
同じくコピー＋チェーン方式で拡張できます。

## 組み込みSurrogateManager

| クラス | 方式 |
|---|---|
| `GlobalSurrogateManager` | アーカイブ全体で1回グローバルにフィットし、全候補を一括predict/score |
| `LocalSurrogateManager` | 候補ごとにk近傍でローカルフィット |
| `CompositeSurrogateManager` | 複数マネージャーのスコアを合成 |
| `PairwiseSurrogateManager` | ペア比較サロゲートによるスコアリング |

`GlobalSurrogateManager(surrogate, acquisition, training_set=None, accuracy_evaluator=None)`は、`training_set`省略時に`ArchiveObjectiveSet()`が使われます。

`LocalSurrogateManager(surrogate, acquisition, training_set=None, accuracy_evaluator=None)`は、`training_set`省略時に`KNNObjectiveSet(n_neighbors=50)`が使われます。
`n_neighbors`は`LocalSurrogateManager`自体のコンストラクタ引数ではなく、既定の`training_set`が持つパラメータです。
候補間で同一の`surrogate`インスタンスを使い回して再フィットする実装のため、スレッドセーフではありません。

`CompositeSurrogateManager(managers, combine_fn)`は、各`managers`の`score_candidates`を独立に呼び、結果のスコア配列を`combine_fn`で合成します。
`combine_fn`に渡す関数として、`product_combine`（要素積。例：EI×PoF）と`rank_weighted_combine(weights=None)`（ランク正規化した重み付き平均を返す関数を生成する）が用意されています。

`PairwiseSurrogateManager(surrogate, training_set=None, n_ref=10)`は、`training_set`省略時に`PairwiseComparisonSet()`が使われます。

各マネージャーの`training_set`/`accuracy_evaluator`引数の詳細は、それぞれ[TrainingSet](training_set.md)と[サロゲート精度評価と動的切り替え](surrogate_switching.md)を参照してください。

### ArchiveBasedManager: サロゲートを学習しない系列

`ArchiveBasedManager`は`SurrogateManager`の抽象サブクラスで、サロゲートモデルを一切学習せず、アーカイブから直接候補をスコアリングします。
抽象メソッドは`compute_scores(candidates_x, archive, ctx=None) -> np.ndarray`だけで、`score_candidates()`は`compute_scores()`の結果を`tell_f=NaN`の`SurrogatePrediction`でラップします。
スコアが目的関数値ではないため、そのまま`tell_f`として使うとpbestなどが汚染されてしまいます。それを防ぐための設計です。

| クラス | パラメータ | スコアの意味 |
|---|---|---|
| `NoveltyManager` | `k=1` | アーカイブへのk近傍平均距離が大きいほど良い |
| `DensityManager` | `eps=1.0` | ε近傍密度の逆数（疎な領域を優先） |
| `NichingManager` | なし | 候補間の最小距離＋アーカイブとの最小距離 |

## 独自SurrogateManagerの実装方法

独自のスコアリング方式が必要な場合は、`SurrogateManager`を継承して`score_candidates()`だけを実装すればよいです。
サロゲートを学習せずアーカイブから直接スコアするパターンであれば、`ArchiveBasedManager`を継承して`compute_scores()`だけを実装するほうが軽量です。

```python
import numpy as np
from saealib import ArchiveBasedManager


class ConstantScoreManager(ArchiveBasedManager):
    """常に一定スコアを返す、archiveを一切参照しない最小限のサロゲートマネージャ。"""

    def compute_scores(self, candidates_x, archive, ctx=None):
        return np.ones(len(candidates_x))
```

## 関連コンポーネント

- [Surrogate](surrogate.md) — `SurrogateManager`が協調させるfit/predictの実体
- [TrainingSet](training_set.md) — 各`SurrogateManager`が学習データの抽出に使う
- [AcquisitionFunction](acquisition_functions.md) — `GlobalSurrogateManager`/`LocalSurrogateManager`の`acquisition`引数
- [サロゲート精度評価と動的切り替え](surrogate_switching.md) — `accuracy_evaluator`/`last_accuracy`の詳細
- [strategies](strategies.md) — `score_candidates()`を呼ぶ側

## 参照

- {py:class}`saealib.SurrogateManager`
- {py:class}`saealib.GlobalSurrogateManager`
- {py:class}`saealib.LocalSurrogateManager`
- {py:class}`saealib.CompositeSurrogateManager`
- {py:class}`saealib.PairwiseSurrogateManager`
- {py:func}`saealib.product_combine`
- {py:func}`saealib.rank_weighted_combine`
- {py:class}`saealib.ArchiveBasedManager`
- {py:class}`saealib.NoveltyManager`
- {py:class}`saealib.DensityManager`
- {py:class}`saealib.NichingManager`
