"use client";

import { useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from "react";

/**
 * Native horizontal scrolling remains available for touch/trackpads.
 * Mouse drag only captures the pointer after a 6px movement, so ordinary
 * clicks on links are not intercepted.  No document-level listeners.
 */
export function useDragToScroll<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const gesture = useRef<{
    pointerId: number;
    x: number;
    y: number;
    scrollLeft: number;
    dragged: boolean;
  } | null>(null);
  const [dragging, setDragging] = useState(false);
  const suppressClick = useRef(false);

  function onPointerDown(event: ReactPointerEvent<T>) {
    if (event.pointerType !== "mouse" || event.button !== 0) return;
    gesture.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      scrollLeft: event.currentTarget.scrollLeft,
      dragged: false,
    };
    suppressClick.current = false;
  }
  function onPointerMove(event: ReactPointerEvent<T>) {
    const current = gesture.current;
    if (!current || current.pointerId !== event.pointerId) return;
    const dx = event.clientX - current.x;
    const dy = event.clientY - current.y;
    if (!current.dragged) {
      if (Math.abs(dx) < 6 || Math.abs(dx) < Math.abs(dy)) return;
      current.dragged = true;
      suppressClick.current = true;
      setDragging(true);
      event.currentTarget.setPointerCapture(event.pointerId);
    }
    event.currentTarget.scrollLeft = current.scrollLeft - dx;
    event.preventDefault();
  }
  function finish(event: ReactPointerEvent<T>) {
    if (gesture.current?.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    gesture.current = null;
    setDragging(false);
  }
  function onClickCapture(event: ReactMouseEvent<T>) {
    if (!suppressClick.current) return;
    event.preventDefault();
    event.stopPropagation();
    suppressClick.current = false;
  }
  return {
    ref,
    "data-dragging": dragging,
    onPointerDown,
    onPointerMove,
    onPointerUp: finish,
    onPointerCancel: finish,
    onClickCapture,
  };
}
