---
primary_layer: cross
---

# 標準インポート

saealibはまだ1.0未満であるため、最初の安定版までは公開APIが変更される可能性があります。この制約のもとで、インポートパスを選ぶ際は次の4層を使います：

## 1. ルートの利便性API

一般的な利用者向けの入口や簡便な例には、ルートパッケージを使います。

```python
from saealib import minimize, Problem, GA
```

ルートのエクスポートは一般的な利用経路を簡単にするためのものです。既存のルートインポートは互換性の一部として維持されますが、別の場所で利用できる名前が自動的にルートのエクスポートになるわけではありません。

## 2. ドメイン名前空間

オブジェクトが特定のドメインに属する場合は公開名前空間を使います。例として`saealib.surrogate`、`saealib.acquisition`、`saealib.operators`があります：

```python
from saealib.surrogate import RBFSurrogate
from saealib.acquisition import MeanPrediction
from saealib.operators import MutationPolynomial
```

SearchSpace、execution、Feedbackの名前は、それぞれ公開名前空間 `saealib.space`、`saealib.execution`、`saealib.policies` からインポートします。

その他の公開ドメイン名前空間も同じ規則に従います。両方が利用できる場合は、より深いモジュールパスより名前空間のエクスポートを優先してください。

## 3. フレームワークとランタイムの拡張API

契約と構成プリミティブのフレームワーク拡張ファサードとして `saealib.core` を使います。

```python
from saealib.core import (
    AssumptionSet,
    Component,
    ComponentContract,
    ComponentGraph,
    DataSpec,
    ExecutionContract,
    LifecycleContract,
    PartSpec,
    PortContract,
    PortSpec,
    StateContract,
)
```

このファサードには、通常のカスタムコンポーネントに必要な契約語彙が含まれます。より専門的な契約記述子は
`saealib.core.contracts` にあります。
`saealib.core.contracts`は互換性専用のパスではなく、公開された専門サブモジュールです。必要な記述子を
`saealib.core`ファサードが公開していない場合は、高度なAPIとして扱ってください。

ランタイムの拡張点には`saealib.execution`を使います。プロバイダー実装者向けの標準的な入口は`RuntimeRegistry`、`RuntimeRegistration`、`create_runtime`です：

```python
from saealib.execution import (
    RuntimeRegistration,
    RuntimeRegistry,
    create_runtime,
)
```

`RuntimeFactory`と`default_runtime_registry`は、ベータ互換性と高度なカスタマイズのためのフックとして残されています。これらは`saealib.execution`から利用できますが、プロバイダー実装者向けの標準的な3名の入口とは異なります。より深い`saealib.execution.runtime`モジュールは実装パスであり、標準的なインポートパスではありません。

非同期プロバイダーの境界では、ノンブロッキングなポーリングの進捗を明示的に報告します。

```python
from saealib.execution import PollResult

return PollResult(state=state, progressed=False)
```

これらのファサードが、フレームワークとランタイムを拡張する際にサポートされる開始点です。saealibが1.0未満である間は、詳細な動作が変わる可能性があります。

## 4. 深い実装パス

公開ファサードより下のパス（`saealib.core.compiler.compiler`など）は、
深い実装パスです。
標準ではなく、互換性も保証されないため、
アプリケーションのインポートや拡張ドキュメントの基盤として使わないでください。
