<!-- DRAFT: machine-assembled from the retired gettext catalogs. Needs a human pass. -->

# Translation guide

`docs/en/` and `docs/ja/` are two independent Markdown trees holding the same set of
pages at the same relative paths. This page records the conventions that keep them
consistent. It is a contributor reference, not a published page — it sits in `docs/`
rather than in either language tree, so no Sphinx build picks it up.

## Which language is authoritative

English is canonical: when the two trees disagree about behaviour, the English page is
correct and the Japanese page is the one to fix.

Canonical is not an authoring order. A change may start in either language. Writing the
Japanese page first and translating it into English afterwards is expected, and so is the
reverse when the change arrives as an English-only contribution.

## Page parity

Every page exists in both trees at the same relative path. `scripts/check_docs_languages.py`
enforces this in CI and admits no exclusions, because `docs/_static/js/lang-switch.js`
rewrites the URL path instead of looking a page up: a page missing from one tree turns the
other tree's language switcher into a 404.

When a page lands in one language only, add a stub at the matching path in the other tree
carrying a short "not translated yet" line and a link to the original, rather than an
exclusion entry.

## What is translated

Prose, headings, table headers, and admonition bodies are translated. The following are
copied verbatim:

- Frontmatter keys and values (`primary_layer`, `related_layers`, `page_type`)
- `python` code blocks, including the comments inside them
- Directive names and options (`{grid-item-card}`, `{note}`, `:link:`, `:link-type:`)
- Link and cross-reference targets
- `toctree` entries
- `{fa}` icon specifications
- API identifiers: class, function, parameter, and attribute names

Content between `<!-- BEGIN GENERATED ... -->` markers is written by a script and must not
be edited or translated by hand in either tree. `scripts/generate_stage_docs.py` regenerates
it for every language; only the table header row is localized there.

## Terminology

Concepts are translated; API identifiers keep their English spelling even when a natural
Japanese rendering exists. `Surrogate` the class stays `Surrogate`; "surrogate model" the
concept becomes サロゲートモデル.

| English | 日本語 |
|---|---|
| surrogate model | サロゲートモデル |
| acquisition function | 獲得関数 |
| surrogate management | サロゲート管理 |
| search algorithm | 探索アルゴリズム |
| parent selection | 親選択 |
| survivor selection | 生存選択 |
| evaluation strategy | 評価戦略 |
| external library adapter | 外部ライブラリアダプタ |
| extension point | 拡張点 |
| invariant | 不変条件 |
| complexity | 計算量 |
| pseudocode | 擬似コード |
| related components | 関連コンポーネント |
| return type | 戻り値の型 |

Recurring section headings: References → 参照, Parameters → パラメータ, Overview → 概要,
Role → 役割, Characteristics → 特徴, Examples → 例, Implementations → 実装.

Class names kept as-is include `Component`, `Crossover`, `Mutation`, `Termination`,
`Runtime`, `Graph`, `SearchSpace`, `Population`, `Problem`, `Decomposition`,
`CallbackManager`, `Evaluator`, `OptimizationStrategy`, and `Surrogate`.

## Formatting

Do not start a line with a space. A continuation line beginning with whitespace is joined
to the previous line with a space inserted, which shows up inside Japanese text.

Line lengths need not match between the trees, but fences and directive open/close pairs
must correspond one to one.
