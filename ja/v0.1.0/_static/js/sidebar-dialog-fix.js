// Workaround for a pydata-sphinx-theme bug (v0.19.0/v0.20.0, not reported
// upstream): resizing past the 960px breakpoint leaves the mobile sidebar
// <dialog> stuck open with its backdrop blocking the page.
document.addEventListener("DOMContentLoaded", function () {
  "use strict";

  var desktopQuery = window.matchMedia("(min-width: 960px)");
  var dialogIds = ["pst-primary-sidebar-modal", "pst-secondary-sidebar-modal"];

  function closeStuckDialogs() {
    dialogIds.forEach(function (id) {
      var dialog = document.getElementById(id);
      if (dialog && dialog.open) {
        dialog.close();
      }
    });
  }

  desktopQuery.addEventListener("change", function (e) {
    if (e.matches) {
      closeStuckDialogs();
    }
  });
});
