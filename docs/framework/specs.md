---
primary_layer: layer4
related_layers: [layer3]
page_type: concept
---

# 宣言要素（Specs）

契約や探索空間の境界では、処理そのものではなく、接続条件や表現を宣言値として渡します。
このページでは、これらの宣言値をまとめて宣言要素と呼びます。
宣言要素はComponentの派生型ではなく、Compilerが接続と要求を判定するための不変値です。

## PortSpecとPortContract

`PortSpec`はポート名、方向、DataSpec、cardinality、要求サービス、optional性を持つ一つのポート宣言です。
`PortContract`は一つのroleに属する入力ポートと出力ポートの集合です。
入力と出力のポート名は、それぞれの方向で一意でなければなりません。

`ONE`、`MANY`、`OPTIONAL`のcardinalityは、提供側の値が消費側の要求を満たすかをCompilerが判定します。
方向、登録済みのdata kind、schema binding、cardinalityが一致しない接続は互換ではありません。

## DataSpecとServiceRequirement

`DataSpec`は登録済みのnominal data kindとschema bindingを表します。
固定値、変数、包含条件、productによるbindingを使って、ポート間で同じスキーマ変数を統合できます。
DataSpecのkind互換性、schemaのunification、サービス解決は別々の検査です。

`ServiceRequirement`は、ポートまたはComponentが必要とする名前付きサービスを宣言します。
`SearchSpace`や`Problem`が提供するサービスと照合され、たとえばFeatureEncoderやSamplingServiceの要求を未解決のまま計画へ進めません。

## RepresentationSpecとの境界

`RepresentationSpec`はSearchSpace側の候補表現仕様です。
候補の種類、形状、表現方法を記述しますが、ComponentContractのports、state、executionには含まれません。
したがって、ポートが要求するFeatureEncoderなどのサービスと、空間が保持するRepresentationSpecは、Compilerが接続する別の境界です。

| 宣言値 | 所有者 | 利用する境界 |
|---|---|---|
| `PortSpec`、`PortContract` | `ComponentContract` | Graphの接続とポート互換性 |
| `DataSpec` | ポートまたは契約 | data kindとschema bindingの統合 |
| `ServiceRequirement` | Componentまたはポート | SearchSpace、Problemが提供するサービスの解決 |
| `RepresentationSpec` | `SearchSpace` | 候補表現と表現サービスの整合性 |

## 生成時点、所有者、拡張

PortSpec、DataSpec、ServiceRequirementはComponentContractを生成するときに作られ、Compilerがグラフ接続時に利用します。
値の所有者は契約またはSearchSpaceであり、Runtimeが実行中に書き換えるものではありません。
新しいデータkindやサービスを追加する場合は、登録規則と互換性規則を明示し、暗黙の型変換で接続しない設計にします。

## 失敗と参照

未登録のkind、未定義のcardinality、schema bindingの不一致、要求サービスの不足はDiagnosticsになります。
ポート宣言の具体的な型と公開import経路は[APIリファレンス](../api/index.md)で確認してください。
候補表現の拡張は[探索空間（SearchSpace）](../concepts/problem_and_ranking/search_space.md)を、Compiler ruleの拡張は[フレームワーク拡張](extensions.md)を参照してください。
