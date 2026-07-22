// Fills the lang-switcher__menu(s) from _templates/lang-switcher.html
// (rendered twice via conf.py's navbar_persistent: desktop + mobile).
document.addEventListener("DOMContentLoaded", function () {
  "use strict";

  var menus = document.querySelectorAll(".lang-switcher__menu");
  if (!menus.length) {
    return;
  }

  var VERSION_SEGMENT = /^(dev|v\d+\.\d+\.\d+)$/;

  var path = window.location.pathname;
  var hasTrailingSlash = path.charAt(path.length - 1) === "/";
  var segments = path.split("/").filter(Boolean);
  var versionIdx = -1;
  for (var i = 0; i < segments.length; i++) {
    if (VERSION_SEGMENT.test(segments[i])) {
      versionIdx = i;
      break;
    }
  }
  if (versionIdx === -1) {
    return;
  }

  var isJa = versionIdx > 0 && segments[versionIdx - 1] === "ja";

  function pageUrlForLanguage(toJa) {
    if (toJa === isJa) {
      return window.location.href;
    }
    var otherSegments;
    if (isJa) {
      otherSegments = segments.slice(0, versionIdx - 1).concat(segments.slice(versionIdx));
    } else {
      otherSegments = segments.slice(0, versionIdx).concat(["ja"], segments.slice(versionIdx));
    }
    var otherPath = "/" + otherSegments.join("/") + (hasTrailingSlash ? "/" : "");
    return window.location.origin + otherPath + window.location.search + window.location.hash;
  }

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // Mirrors the theme's version-switcher item markup so custom.css's
  // .active span:before rule applies here too.
  function dropdownItem(label, url, current) {
    return (
      '<a class="dropdown-item list-group-item list-group-item-action py-1' +
      (current ? " active" : "") +
      '" href="' + escapeHtml(url) + '" role="option"' +
      "><span>" + escapeHtml(label) + "</span></a>"
    );
  }

  var itemsHtml =
    dropdownItem("English", pageUrlForLanguage(false), !isJa) +
    dropdownItem("日本語", pageUrlForLanguage(true), isJa);
  var currentLabel = isJa ? "日本語" : "English";

  menus.forEach(function (menu) {
    menu.innerHTML = itemsHtml;
  });
  document.querySelectorAll(".lang-switcher__button").forEach(function (btn) {
    // Drops the caret span, matching the theme's own version-switcher JS.
    btn.textContent = currentLabel;
  });
});
