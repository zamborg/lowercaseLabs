import React, { useEffect, useRef } from "react";
import { CHAT_PANE } from "../core/contracts";

export function ChatDivider({
  width,
  max,
  onWidth,
  onCommit,
  onDragging,
}: {
  width: number;
  max: number;
  onWidth: (width: number) => void;
  onCommit: (width: number) => void;
  onDragging: (dragging: boolean) => void;
}) {
  const stopDrag = useRef<(() => void) | undefined>(undefined);
  useEffect(() => () => stopDrag.current?.(), []);
  const clamp = (value: number) =>
    Math.max(CHAT_PANE.min, Math.min(max, Math.round(value)));
  function beginDrag(event: React.PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || stopDrag.current) return;
    event.preventDefault();
    event.currentTarget.focus();
    const { pointerId, clientX } = event;
    let latest = width;
    function move(next: PointerEvent) {
      if (next.pointerId !== pointerId) return;
      latest = clamp(width + next.clientX - clientX);
      onWidth(latest);
    }
    function release(next: PointerEvent) {
      if (next.pointerId === pointerId) finish();
    }
    function finish() {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", release);
      window.removeEventListener("pointercancel", release);
      window.removeEventListener("blur", finish);
      stopDrag.current = undefined;
      onDragging(false);
      if (latest !== width) onCommit(latest);
    }
    stopDrag.current = finish;
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", release);
    window.addEventListener("pointercancel", release);
    window.addEventListener("blur", finish);
    onDragging(true);
  }
  return (
    <div
      className="chat-divider"
      role="separator"
      aria-label="Resize chat pane"
      aria-controls="chat-pane"
      aria-orientation="vertical"
      aria-valuemin={CHAT_PANE.min}
      aria-valuemax={max}
      aria-valuenow={width}
      tabIndex={0}
      onPointerDown={beginDrag}
      onDoubleClick={() => onCommit(clamp(CHAT_PANE.initial))}
      onKeyDown={(event) => {
        const next = {
          ArrowLeft: width - 24,
          ArrowRight: width + 24,
          Home: CHAT_PANE.min,
          End: max,
        }[event.key];
        if (next === undefined) return;
        event.preventDefault();
        onCommit(clamp(next));
      }}
    />
  );
}
