/**
 * Sync Mermaid SVG rendering width to its native viewBox width,
 * ensuring 1:1 scale matching document body text font size exactly.
 */
function syncMermaidScale() {
    const svgs = document.querySelectorAll("div.mermaid svg, pre.mermaid > svg, .mermaid-container > pre > svg");
    svgs.forEach((svg) => {
        const viewBox = svg.getAttribute("viewBox");
        if (viewBox) {
            const parts = viewBox.trim().split(/[\s,]+/);
            if (parts.length === 4) {
                const origWidth = parseFloat(parts[2]);
                const origHeight = parseFloat(parts[3]);
                if (origWidth > 0 && origHeight > 0) {
                    // Set SVG style width & height to viewBox native pixels (scale = 1.0)
                    svg.style.setProperty("width", `${origWidth}px`, "important");
                    svg.style.setProperty("height", `${origHeight}px`, "important");
                    svg.style.setProperty("max-width", "none", "important");
                }
            }
        }
    });
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
        syncMermaidScale();
        setTimeout(syncMermaidScale, 300);
        setTimeout(syncMermaidScale, 1000);
    });
} else {
    syncMermaidScale();
    setTimeout(syncMermaidScale, 300);
    setTimeout(syncMermaidScale, 1000);
}

// Observe dynamic Mermaid insertion / details expansion
const observer = new MutationObserver(() => {
    syncMermaidScale();
});
observer.observe(document.body, { childList: true, subtree: true });
