"""Browser leave/stop confirmation while profile generation is running.

Streamlit runs ``components.html`` inside an iframe. This module escapes that
iframe by injecting a ``<script>`` into the parent Streamlit document so click /
unload handlers actually see the app UI.

When the user confirms they want to stop, ``window.confirm`` has already consumed
the original gesture — so we explicitly stop the Streamlit run and re-dispatch
the intended action.
"""

from __future__ import annotations

import json
import time

import streamlit.components.v1 as components

_CONFIRM_MESSAGE = (
    "Are you sure you would like to stop the template generation?"
)

# Streamlit must deliver the iframe + parent script before generation blocks.
_ENABLE_FLUSH_SECONDS = 0.75


def _parent_guard_js(active: bool) -> str:
    """JavaScript that runs in the Streamlit page (not the component iframe)."""
    return f"""
(function () {{
  var CONFIRM_MSG = {json.dumps(_CONFIRM_MESSAGE)};
  var ACTIVE = {json.dumps(active)};
  var w = window;
  var doc = document;

  w.__aamGenerating = ACTIVE;
  try {{
    if (ACTIVE) doc.documentElement.setAttribute("data-aam-generating", "true");
    else doc.documentElement.removeAttribute("data-aam-generating");
  }} catch (e) {{}}

  function isActive() {{
    if (w.__aamBypass) return false;
    return w.__aamGenerating === true ||
      doc.documentElement.getAttribute("data-aam-generating") === "true";
  }}

  function ask() {{
    return w.confirm(CONFIRM_MSG);
  }}

  function disarm() {{
    w.__aamGenerating = false;
    try {{ doc.documentElement.removeAttribute("data-aam-generating"); }} catch (e) {{}}
  }}

  function labelOf(el) {{
    if (!el) return "";
    return ((el.getAttribute && (el.getAttribute("aria-label") || el.getAttribute("title"))) ||
      (el.textContent || "")).trim().toLowerCase();
  }}

  function inUploader(el) {{
    return !!(el && el.closest && el.closest(
      '[data-testid="stFileUploader"],[data-testid="stFileUploaderDropzone"],' +
      '[data-testid="stFileUploaderDropzoneInput"],[data-testid="stFileUploaderDeleteBtn"],' +
      '[data-testid="stFileUploaderFile"]'
    ));
  }}

  function isRisky(el) {{
    if (!el || !el.closest) return false;
    if (inUploader(el)) return true;
    if (el.closest(
      '[data-testid="stDownloadButton"],[data-testid="stExpander"],' +
      '[data-testid="stExpanderToggleIcon"],[data-testid="stButton"]'
    )) return true;
    var btn = el.closest("button, [role='button']");
    if (btn) {{
      var label = labelOf(btn);
      if (label === "stop" || label.indexOf("stop ") === 0) return true;
      if (label === "rerun" || label.indexOf("rerun") === 0) return true;
      if (btn.closest('[data-testid="stMain"], [data-testid="stAppViewContainer"],' +
                      '[data-testid="stToolbar"], [data-testid="stAppToolbar"]')) {{
        return true;
      }}
    }}
    var a = el.closest("a[href]");
    if (a && a.getAttribute("href") && a.getAttribute("href") !== "#" &&
        !a.hasAttribute("download") &&
        a.getAttribute("href").indexOf("javascript:") !== 0) {{
      return true;
    }}
    return false;
  }}

  function block(e) {{
    e.preventDefault();
    e.stopPropagation();
    if (e.stopImmediatePropagation) e.stopImmediatePropagation();
  }}

  function findStopButton() {{
    var nodes = doc.querySelectorAll("button, [role='button']");
    for (var i = 0; i < nodes.length; i++) {{
      var label = labelOf(nodes[i]);
      if (label === "stop" || label.indexOf("stop ") === 0) return nodes[i];
    }}
    return null;
  }}

  function actionableFrom(target) {{
    if (!target || !target.closest) return target;
    return target.closest(
      'button, [role="button"], a[href], input[type="file"],' +
      '[data-testid="stFileUploaderDeleteBtn"],' +
      '[data-testid="stFileUploaderDropzone"],' +
      '[data-testid="stFileUploaderDropzoneInput"],' +
      '[data-testid="stButton"]'
    ) || target;
  }}

  function stopThenReplay(target) {{
    // confirm() already consumed the user gesture; Streamlit never saw it.
    disarm();
    var stopBtn = findStopButton();
    if (stopBtn) {{
      w.__aamBypass = true;
      try {{ stopBtn.click(); }} catch (e) {{}}
      w.__aamBypass = false;
    }}
    var el = actionableFrom(target);
    setTimeout(function () {{
      w.__aamBypass = true;
      try {{
        if (el && typeof el.click === "function") el.click();
      }} catch (e) {{}}
      setTimeout(function () {{ w.__aamBypass = false; }}, 250);
    }}, 150);
  }}

  function onPointer(e) {{
    if (!isActive()) return;
    if (!isRisky(e.target)) return;
    if (!ask()) {{
      block(e);
      return;
    }}
    block(e);
    if (inUploader(e.target)) {{
      w.__aamAllowFileChange = true;
      clearTimeout(w.__aamAllowFileTimer);
      w.__aamAllowFileTimer = setTimeout(function () {{
        w.__aamAllowFileChange = false;
      }}, 60000);
    }}
    stopThenReplay(e.target);
  }}

  function onChange(e) {{
    if (!isActive()) return;
    var t = e.target;
    if (!(inUploader(t) || (t && t.type === "file"))) return;
    if (w.__aamAllowFileChange) {{
      w.__aamAllowFileChange = false;
      return;
    }}
    if (!ask()) {{
      block(e);
      try {{ t.value = ""; }} catch (err) {{}}
      return;
    }}
    // Confirmed — allow this change through and stop the running script.
    disarm();
    var stopBtn = findStopButton();
    if (stopBtn) {{
      w.__aamBypass = true;
      try {{ stopBtn.click(); }} catch (err) {{}}
      w.__aamBypass = false;
    }}
  }}

  function onDrop(e) {{
    if (!isActive()) return;
    if (!inUploader(e.target)) return;
    if (!ask()) {{
      block(e);
      return;
    }}
    block(e);
    w.__aamAllowFileChange = true;
    stopThenReplay(e.target);
  }}

  function onDragOver(e) {{
    if (!isActive()) return;
    if (!inUploader(e.target)) return;
    e.preventDefault();
  }}

  function onKey(e) {{
    if (!isActive()) return;
    if (e.key !== "Enter" && e.key !== " ") return;
    if (!isRisky(e.target)) return;
    if (!ask()) {{
      block(e);
      return;
    }}
    block(e);
    stopThenReplay(e.target);
  }}

  function onBeforeUnload(e) {{
    if (!isActive()) return;
    e.preventDefault();
    e.returnValue = CONFIRM_MSG;
    return CONFIRM_MSG;
  }}

  var HANDLER_VERSION = 2;
  if (w.__aamLeaveGuardVersion !== HANDLER_VERSION) {{
    var prev = w.__aamLeaveGuardHandlers;
    if (prev) {{
      try {{
        doc.removeEventListener("pointerdown", prev.onPointer, true);
        doc.removeEventListener("change", prev.onChange, true);
        doc.removeEventListener("drop", prev.onDrop, true);
        doc.removeEventListener("dragover", prev.onDragOver, true);
        doc.removeEventListener("keydown", prev.onKey, true);
        w.removeEventListener("beforeunload", prev.onBeforeUnload);
      }} catch (e) {{}}
    }}
    var handlers = {{
      onPointer: onPointer,
      onChange: onChange,
      onDrop: onDrop,
      onDragOver: onDragOver,
      onKey: onKey,
      onBeforeUnload: onBeforeUnload
    }};
    w.__aamLeaveGuardHandlers = handlers;
    w.__aamLeaveGuardVersion = HANDLER_VERSION;
    w.__aamLeaveGuardBound = true;
    doc.addEventListener("pointerdown", onPointer, true);
    doc.addEventListener("change", onChange, true);
    doc.addEventListener("drop", onDrop, true);
    doc.addEventListener("dragover", onDragOver, true);
    doc.addEventListener("keydown", onKey, true);
    w.addEventListener("beforeunload", onBeforeUnload);
  }}
}})();
"""


def _iframe_bootstrap(active: bool) -> str:
    """Iframe script that installs/updates the parent-page guard script."""
    parent_js = json.dumps(_parent_guard_js(active))
    return f"""
<script>
(function () {{
  var BOOT_ID = "aam-leave-guard-boot";
  var parentCode = {parent_js};

  function findAppDocument() {{
    var tried = [];
    var w = window;
    try {{
      while (w) {{
        tried.push(w);
        if (!w.parent || w.parent === w) break;
        w = w.parent;
      }}
    }} catch (e) {{}}
    for (var i = 0; i < tried.length; i++) {{
      try {{
        var d = tried[i].document;
        if (d && (d.querySelector('[data-testid="stApp"]') || d.querySelector(".stApp"))) {{
          return {{ win: tried[i], doc: d }};
        }}
      }} catch (e) {{}}
    }}
    try {{
      return {{ win: window.parent, doc: window.parent.document }};
    }} catch (e) {{
      return {{ win: window, doc: document }};
    }}
  }}

  var ctx = findAppDocument();
  var appDoc = ctx.doc;
  var old = appDoc.getElementById(BOOT_ID);
  if (old) old.remove();
  var s = appDoc.createElement("script");
  s.id = BOOT_ID;
  s.text = parentCode;
  (appDoc.head || appDoc.documentElement).appendChild(s);
}})();
</script>
"""


def set_generation_leave_guard(active: bool, *, flush: bool = False) -> None:
    """Enable or disable interruption confirmation in the browser.

    Popups only appear while ``active`` is True (generation in progress).
    """
    components.html(_iframe_bootstrap(active), height=0)
    if active and flush:
        time.sleep(_ENABLE_FLUSH_SECONDS)
