from docutils import nodes


_LAYER_NAMES = {
    "layer1": "Layer 1: Use",
    "layer2": "Layer 2: Compose",
    "layer3": "Layer 3: Extend Components",
    "layer4": "Layer 4: Extend Framework",
    "cross": "Cross-layer",
}


def _add_layer_badge(app, doctree, docname):
    if app.builder.name != "html":
        return
    if docname.rstrip("/") in {"", "index"}:
        return

    metadata = app.env.metadata.get(docname, {})
    primary_layer = metadata.get("primary_layer")
    if primary_layer not in _LAYER_NAMES:
        return

    related_layers = metadata.get("related_layers", [])
    if not isinstance(related_layers, (list, tuple)):
        related_layers = []

    layers = []
    for layer in [primary_layer, *related_layers]:
        if layer not in _LAYER_NAMES or layer in layers:
            continue
        layers.append(layer)

    if not layers:
        return

    title = next(
        (node for node in doctree.traverse(nodes.title) if isinstance(node.parent, nodes.section)),
        None,
    )
    if title is None:
        return

    badges = "".join(
        f'<span class="layer-badge">{_LAYER_NAMES[layer]}</span>' for layer in layers
    )
    badge_html = f'<div class="layer-badges">{badges}</div>'
    badge = nodes.raw("", badge_html, format="html")
    title.parent.insert(title.parent.index(title) + 1, badge)


def setup(app):
    app.connect("doctree-resolved", _add_layer_badge)
    return {"version": "1", "parallel_read_safe": True}
