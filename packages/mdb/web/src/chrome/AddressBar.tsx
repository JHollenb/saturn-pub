// The cursor address, editable. `g` focuses it. Enter navigates; the server parses the text.
import { useEffect, useRef, useState } from 'react';
import { useCursor } from '../state/cursor';
import { useUi } from '../state/ui';

export const ADDRESS_INPUT_ID = 'mdb-address';

export function AddressBar() {
  const addr = useCursor((s) => s.addr);
  const go = useCursor((s) => s.go);
  const back = useCursor((s) => s.back);
  const forward = useCursor((s) => s.forward);
  const index = useCursor((s) => s.index);
  const length = useCursor((s) => s.history.length);
  const set = useUi((s) => s.set);
  const [text, setText] = useState(addr ?? '');
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => setText(addr ?? ''), [addr]);
  return (
    <form
      className="address-bar"
      onSubmit={(e) => {
        e.preventDefault();
        const next = text.trim();
        if (next) go(next);
        set({ inspect: null });
        ref.current?.blur();
      }}
    >
      <button
        type="button"
        onClick={back}
        disabled={index <= 0}
        title="cursor history back (Alt+←)"
      >
        ‹
      </button>
      <button
        type="button"
        onClick={forward}
        disabled={index >= length - 1}
        title="cursor history forward (Alt+→)"
      >
        ›
      </button>
      <input
        id={ADDRESS_INPUT_ID}
        ref={ref}
        className="mono address-input"
        value={text}
        spellCheck={false}
        aria-label="occurrence address"
        placeholder="address, e.g. t2.L1.h"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') {
            setText(addr ?? '');
            ref.current?.blur();
          }
        }}
      />
    </form>
  );
}
