---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# コンパイラ（Compiler）

`ComponentGraph`を解決して検証すると、Compilerは実行から分離された`ExecutablePlan`を返します。
CompilerはRuntimeを実行せず、Runtimeが必要とする能力と診断を計画へ記録します。

## コンパイル、検証、解決

Compilation全体は、契約スナップショット取得、Resolution、Verificationの順に進みます。
`ResolutionRule`はサービス、Adapter、schema bindingなどの解決候補を、claimしたGraph位置への提案として返します。
`VerificationRule`は解決済みGraphを観察し、変更せずにDiagnosticsを返します。
共通の`CompilationRule`はnamespace、name、phase、`apply(RuleContext)`の境界を持ちます。

Resolutionは提案が収束するまで反復され、同じGraph位置への競合や未申告の変更はエラー診断になります。
VerificationはGraphのwell-formedness、ポート互換性、サービス、データフロー、状態効果、ライフサイクル、Runtime capabilityを確認します。

## 診断（Diagnostics）

`Diagnostic`はseverity、code、message、contract path、resolution hintを持つ検証結果です。
Compilerは失敗を任意の例外だけで表さず、複数の問題をDiagnosticBagへ集めて計画の判断材料にします。
代表的な失敗は、未知のノード端点、未接続の必須ポート、DataSpecの不一致、要求サービスの不足、状態の未宣言アクセス、Runtime capability不足です。

## 実行計画（ExecutablePlan）

`ExecutablePlan`は解決済みGraph、Diagnostics、必要なRuntime capability、有効なrule、挿入されたAdapter、契約スナップショットを保持する不変値です。
計画は実行位置や現在の状態を所有しないため、同じ計画をRuntimeSessionへ渡して実行と再開を分離できます。
Diagnosticsにエラーが残る場合、その計画を実行可能な入力として扱うかは呼び出し側の検査とRuntimeの契約に従います。

## 検証から実行まで

```{mermaid}
flowchart LR
    C[Component] --> N[ComponentNode]
    N --> G[ComponentGraph]
    G --> K[Compiler]
    K --> P[ExecutablePlan]
    P --> R[ExecutionRuntime]
```

ComponentからGraphまでは構成、Compilerは解決と検証、ExecutablePlanから先はRuntimeの責務です。
RuntimeはPlanのノード結果からStatePatch、イベント、コマンドを適用し、検証済みの状態境界を次のStepへ渡します。

## 拡張時の不変条件

Compiler ruleは宣言されていないGraph位置を変更せず、StructuredRegionの実行木を解決規則で勝手に変更しません。
Adapterを使う場合も、変換の意味とlossless性を明示し、ポート互換性の検査を迂回しません。
Compiler ruleの型やimport経路はリリースごとの[APIリファレンス](../api/index.md)で確認してください。
拡張方針は[フレームワーク拡張](extensions.md)を参照してください。
