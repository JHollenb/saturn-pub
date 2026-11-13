// Global keys. Arrows walk the recorded history (frame prev/next); Alt+arrows walk the cursor
// history; g focuses the address bar; / opens the command line; ? toggles help.
// Rules: text inputs keep their arrows and letters; a widget that owns its keys (marked
// `data-own-keys`, e.g. the timeline grid) keeps them too; any Cmd/Ctrl/Meta combination is left
// to the browser; an event another handler already used is not acted on twice.
import { useEffect } from 'react';
import type { Frame } from '../api/types.gen';
import { ADDRESS_INPUT_ID } from '../chrome/AddressBar';
import { useCursor } from '../state/cursor';
import { useUi } from '../state/ui';

export interface KeyLike {
  key: string;
  altKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  shiftKey: boolean;
  defaultPrevented: boolean;
  isComposing?: boolean;
  target: EventTarget | null;
}

export type KeyAction =
  | { kind: 'close' }
  | { kind: 'back' }
  | { kind: 'forward' }
  | { kind: 'go'; addr: string }
  | { kind: 'focus-address' }
  | { kind: 'command' }
  | { kind: 'help' };

interface ElementLike {
  tagName?: string;
  isContentEditable?: boolean;
  closest?: (selector: string) => unknown;
}

/** True when the event target keeps its own keys (text field, select, editable, or a widget
 *  marked `data-own-keys`). Duck-typed so it needs no DOM globals. */
export function typing(target: EventTarget | null): boolean {
  const el = target as ElementLike | null;
  if (!el || typeof el.tagName !== 'string') return false;
  return (
    el.isContentEditable === true ||
    ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName) ||
    (typeof el.closest === 'function' && el.closest('[data-own-keys]') !== null)
  );
}

/** Which action a key press means, or null to leave it alone (and not preventDefault). */
export function keyAction(
  e: KeyLike,
  frame: Pick<Frame, 'prev' | 'next'> | null,
): KeyAction | null {
  if (e.defaultPrevented || e.isComposing) return null;
  if (e.key === 'Escape') return { kind: 'close' };
  if (e.metaKey || e.ctrlKey || typing(e.target)) return null;
  if (e.altKey) {
    if (e.key === 'ArrowLeft') return { kind: 'back' };
    if (e.key === 'ArrowRight') return { kind: 'forward' };
    return null;
  }
  if ((e.key === 'ArrowLeft' || e.key === 'ArrowUp') && frame?.prev)
    return { kind: 'go', addr: frame.prev };
  if ((e.key === 'ArrowRight' || e.key === 'ArrowDown') && frame?.next)
    return { kind: 'go', addr: frame.next };
  if (e.key === 'g') return { kind: 'focus-address' };
  if (e.key === '/') return { kind: 'command' };
  if (e.key === '?') return { kind: 'help' };
  return null;
}

export function useKeys(frame: Frame | null) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const action = keyAction(e, frame);
      if (!action) return;
      const ui = useUi.getState();
      const cursor = useCursor.getState();
      switch (action.kind) {
        case 'close':
          ui.set({ help: false, command: false });
          return;
        case 'back':
          e.preventDefault();
          cursor.back();
          return;
        case 'forward':
          e.preventDefault();
          cursor.forward();
          return;
        case 'go':
          e.preventDefault();
          cursor.go(action.addr);
          return;
        case 'focus-address':
          e.preventDefault();
          document.getElementById(ADDRESS_INPUT_ID)?.focus();
          return;
        case 'command':
          e.preventDefault();
          ui.set({ command: true });
          return;
        case 'help':
          ui.set({ help: !ui.help });
          return;
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [frame]);
}
