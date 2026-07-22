# 多目的最適化

目的間にトレードオフのある多目的最適化問題を、`saealib`で解きます。

アルゴリズム、サロゲート、評価戦略の切り替え方は、目的数によらず[単目的最適化](single_objective.md)の「コンポーネントの切り替え」と共通です。

このページでは、目的数が2以上のときに固有の設定である、比較演算子の選択とパレートフロントの抽出を扱います。

## 問題設定

複数の目的関数がトレードオフの関係にあるとき、一方を改善するともう一方が悪化する解が存在します。

この関係の下で、他のどの解にも全目的で優越されない解の集合を**パレートフロント**と呼びます。

ここでは例として、`saealib`に組み込まれたZDT1関数を最小化します。

```python
from saealib.benchmarks import zdt1

problem = zdt1(n_var=10)
```

`zdt1`は、凸形のパレートフロントを持つ2目的のベンチマーク問題を返す`Problem`インスタンスです。

## 高レベルAPI: minimize

`Problem`インスタンスを直接渡すと、目的数はそこから引き継がれます。

```python
from saealib import minimize

result = minimize(problem, max_fe=2000, seed=0)

print(result.x.shape)  # (n_pareto, dim)
print(result.f.shape)  # (n_pareto, n_obj)
```

単目的では1点だった`result.x`/`result.f`は、多目的ではパレートフロントを構成する複数の解になります。

## 比較演算子の選択

多目的では、候補解同士の優劣を`Comparator`が決めます。

`Problem`の`comparator`引数を省略すると、目的数に応じて自動選択されます（`n_obj == 1`なら`SingleObjectiveComparator`、`n_obj > 1`なら`NSGA2Comparator`）。

| クラス | 動作 |
|--------|------|
| `NSGA2Comparator` | 非優越ソートと混雑度による多様性維持（デフォルト） |
| `SPEA2Comparator` | 優越関係の強さと近傍密度によるフィットネス |
| `HypervolumeComparator` | ハイパーボリューム貢献度による優劣判定 |
| `EpsilonDominanceComparator` | εドミナンスによる優越判定 |
| `NSGA3Comparator` | 参照点による多様性維持。`reference_points`が必須 |
| `RNSGA2Comparator` | 指定した参照点の近傍へ解を集中させる。`reference_points`が必須 |

`comparator`は`Problem`インスタンスの属性として差し替えられます。

```python
from saealib.comparators import HypervolumeComparator

problem.comparator = HypervolumeComparator()
result = minimize(problem, max_fe=2000, seed=0)
```

## パレートフロントの抽出

実行後、`result.ctx.pareto_archive`には最終的なパレートフロントが保持されています。

```python
pareto_x = result.ctx.pareto_archive.get_array("x")
pareto_f = result.ctx.pareto_archive.get_array("f")
```

任意の目的値配列からパレートフロントを求めたい場合は、`non_dominated_sort`を直接使えます。

```python
from saealib.comparators import non_dominated_sort

archive_f = result.ctx.archive.get_array("f")
ranks, fronts = non_dominated_sort(archive_f, direction=problem.direction)
front0_f = archive_f[fronts[0]]  # first non-dominated front
```

## 参照

- {py:func}`saealib.minimize`
- {py:class}`saealib.Problem`
- {py:class}`saealib.NSGA2Comparator` / {py:class}`saealib.SPEA2Comparator` / {py:class}`saealib.HypervolumeComparator` / {py:class}`saealib.EpsilonDominanceComparator` / {py:class}`saealib.NSGA3Comparator` / {py:class}`saealib.RNSGA2Comparator`
- {py:func}`saealib.non_dominated_sort`
