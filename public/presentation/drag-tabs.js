// Presentation tabs: pointer drag without a visible scrollbar.
// An actual click still selects a slide; a drag cannot trigger a click.
(() => {
  const tabs = document.getElementById("slide-tabs");
  if (!tabs) return;
  let gesture = null;
  let blockClick = false;
  tabs.addEventListener("pointerdown", (event) => {
    if (event.pointerType !== "mouse" || event.button !== 0) return;
    gesture = { id: event.pointerId, x: event.clientX, y: event.clientY,
      scroll: tabs.scrollLeft, dragging: false };
    blockClick = false;
  });
  tabs.addEventListener("pointermove", (event) => {
    if (!gesture || event.pointerId !== gesture.id) return;
    const dx = event.clientX - gesture.x;
    const dy = event.clientY - gesture.y;
    if (!gesture.dragging) {
      if (Math.abs(dx) < 6 || Math.abs(dx) < Math.abs(dy)) return;
      gesture.dragging = true;
      blockClick = true;
      tabs.dataset.dragging = "true";
      tabs.setPointerCapture(event.pointerId);
    }
    tabs.scrollLeft = gesture.scroll - dx;
    event.preventDefault();
  });
  const stop = (event) => {
    if (!gesture || event.pointerId !== gesture.id) return;
    if (tabs.hasPointerCapture(event.pointerId)) tabs.releasePointerCapture(event.pointerId);
    gesture = null;
    delete tabs.dataset.dragging;
  };
  tabs.addEventListener("pointerup", stop);
  tabs.addEventListener("pointercancel", stop);
  tabs.addEventListener("click", (event) => {
    if (!blockClick) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    blockClick = false;
  }, true);
})();
