import { describe, expect, it } from 'vitest';
import { composeAddress, formatFragment, originOf, parseFragment } from '../api/address';
import type { ClockAxis } from '../api/types.gen';
import { createCursorStore, type LocationPort } from './cursor';
import { useUi } from './ui';
import { useCursor } from './cursor';
import type { PointerRef } from '../api/types.gen';

function fakeLocation(initial = '') {
  let hash = initial;
  const listeners: (() => void)[] = [];
  const writes: string[] = [];
  const port: LocationPort = {
    read: () => hash,
    write: (h) => {
      hash = h;
      writes.push(h);
    },
    listen: (fn) => {
      listeners.push(fn);
      return () => undefined;
    },
  };
  return {
    port,
    writes,
    /** Simulate the user editing the URL / pressing browser back. */
    navigate(h: string) {
      hash = h;
      listeners.forEach((fn) => fn());
    },
  };
}

const TRICKY = [
  't3.L1.h',
  'b2:t3.L1.h_out',
  '@cut-01:t0.emb',
  't2.L1.p[h3,src=1]',
  't1.L0.h~program',
  't1.L0.h[" hot"=5012]~prog-2',
  't4.logits[" a/b#c%"=17]',
  'tokens[" hot"=5012]',
];

describe('URL fragment codec', () => {
  it.each(TRICKY)('round-trips %s', (addr) => {
    const hash = formatFragment({ sid: 'tiny/run 1', addr });
    expect(hash.startsWith('#/')).toBe(true);
    // The address is one path segment: no raw '/', '#', '%' or space survives encoding.
    expect(hash.slice(2).split('/').length).toBe(2);
    expect(parseFragment(hash)).toEqual({ sid: 'tiny/run 1', addr, record: null, symbol: null });
  });

  it.each(TRICKY)('keeps the address byte-exact beside record and symbol: %s', (addr) => {
    const record = 'rec/1?x=y&z#w';
    const symbol = 'demo.carrier';
    const hash = formatFragment({ sid: 'tiny', addr, record, symbol });
    expect(hash.slice(0, hash.indexOf('?'))).toBe(formatFragment({ sid: 'tiny', addr }));
    expect(parseFragment(hash)).toEqual({ sid: 'tiny', addr, record, symbol });
    expect(parseFragment(formatFragment({ sid: 'tiny', addr, symbol }))).toEqual({
      sid: 'tiny',
      addr,
      record: null,
      symbol,
    });
  });

  it('writes the detail after the address in the documented shape', () => {
    expect(
      formatFragment({
        sid: 'tiny',
        addr: 't1.L0.h[" hot"=5012]~program',
        record: 'r1',
        symbol: 's1',
      }),
    ).toBe('#/tiny/t1.L0.h%5B%22%20hot%22%3D5012%5D~program?record=r1&symbol=s1');
    expect(parseFragment('#/tiny?record=r1')).toEqual({
      sid: 'tiny',
      addr: null,
      record: 'r1',
      symbol: null,
    });
  });

  it('keeps a session with no address', () => {
    expect(parseFragment(formatFragment({ sid: 'qwen', addr: null }))).toEqual({
      sid: 'qwen',
      addr: null,
      record: null,
      symbol: null,
    });
    expect(parseFragment('')).toBeNull();
    expect(parseFragment('#/%E0%A4%A')).toBeNull();
  });

  it('splits origins but not colons inside quoted selectors', () => {
    expect(originOf('b2:t3.L1.h')).toBe('b2:');
    expect(originOf('@abc:t3.L1.h')).toBe('@abc:');
    expect(originOf('t1.L0.h[" a:b"=5]')).toBe('');
  });

  it('composes cell addresses only from server tick fragments', () => {
    const axes: ClockAxis[] = [
      {
        id: 'row',
        label: 'row',
        kind: 'ordinal',
        note: null,
        ticks: [
          {
            i: 0,
            label: '0',
            frag: 't0',
            port: null,
            detail: null,
            addressable: true,
            inferred: false,
          },
        ],
      },
      {
        id: 'stage',
        label: 'stage',
        kind: 'ordinal',
        note: null,
        ticks: [
          {
            i: 0,
            label: 'pre-embed',
            frag: '',
            port: null,
            detail: null,
            addressable: false,
            inferred: false,
          },
          {
            i: 1,
            label: 'L0',
            frag: 'L0',
            port: 'h',
            detail: null,
            addressable: true,
            inferred: false,
          },
        ],
      },
    ];
    expect(composeAddress(axes, 0, 0)).toBeNull(); // no occurrence address names this boundary
    expect(composeAddress(axes, 0, 1, 'b1:')).toBe('b1:t0.L0.h');
    expect(() => composeAddress(axes, 1, 0)).toThrow();
  });
});

describe('cursor store', () => {
  it('mirrors the cursor into the URL and keeps a history', () => {
    const loc = fakeLocation();
    const store = createCursorStore(loc.port);
    store.getState().setSession('tiny', 't0.L0.h');
    store.getState().go('t1.L0.h');
    store.getState().go('t1.L0.h[" hot"=5012]~program');
    expect(loc.port.read()).toBe(
      formatFragment({ sid: 'tiny', addr: 't1.L0.h[" hot"=5012]~program' }),
    );
    expect(store.getState().history).toEqual([
      't0.L0.h',
      't1.L0.h',
      't1.L0.h[" hot"=5012]~program',
    ]);
    store.getState().back();
    expect(store.getState().addr).toBe('t1.L0.h');
    expect(loc.port.read()).toBe(formatFragment({ sid: 'tiny', addr: 't1.L0.h' }));
    // A new go() after back() drops the forward history.
    store.getState().go('t2.L0.h');
    expect(store.getState().history).toEqual(['t0.L0.h', 't1.L0.h', 't2.L0.h']);
    store.getState().forward();
    expect(store.getState().addr).toBe('t2.L0.h');
  });

  it('does not push a duplicate entry for the same address', () => {
    const loc = fakeLocation();
    const store = createCursorStore(loc.port);
    store.getState().setSession('tiny', 't0.L0.h');
    store.getState().go('t0.L0.h');
    expect(store.getState().history).toEqual(['t0.L0.h']);
  });

  it('follows the URL when the user navigates', () => {
    const loc = fakeLocation(formatFragment({ sid: 'tiny', addr: 't0.L0.h' }));
    const store = createCursorStore(loc.port);
    store.getState().syncFromLocation();
    expect(store.getState()).toMatchObject({ sid: 'tiny', addr: 't0.L0.h' });
    loc.navigate(formatFragment({ sid: 'tiny', addr: 'b1:t2.L1.h' }));
    expect(store.getState().addr).toBe('b1:t2.L1.h');
    loc.navigate(formatFragment({ sid: 'qwen', addr: 't2.L1.p' }));
    expect(store.getState()).toMatchObject({ sid: 'qwen', addr: 't2.L1.p', history: ['t2.L1.p'] });
    // A URL that already equals the cursor is not written again.
    const before = loc.writes.length;
    store.getState().syncFromLocation();
    expect(loc.writes.length).toBe(before);
  });
});

describe('record and symbol detail in the URL', () => {
  it('writes opened detail into the fragment and reads it back from a pasted link', () => {
    const loc = fakeLocation();
    const store = createCursorStore(loc.port);
    store.getState().setSession('tiny', 't2.L0.h[" hot"=5012]~program');
    store.getState().open({ record: 'rec-9', symbol: 'demo.carrier' });
    expect(loc.port.read()).toBe(
      formatFragment({
        sid: 'tiny',
        addr: 't2.L0.h[" hot"=5012]~program',
        record: 'rec-9',
        symbol: 'demo.carrier',
      }),
    );
    store.getState().open({ record: null });
    expect(loc.port.read()).not.toContain('record=');
    expect(store.getState().symbol).toBe('demo.carrier');

    const pasted = fakeLocation(
      formatFragment({ sid: 'qwen', addr: 't2.L1.p', record: 'r', symbol: 's' }),
    );
    const other = createCursorStore(pasted.port);
    other.getState().syncFromLocation();
    expect(other.getState()).toMatchObject({
      sid: 'qwen',
      addr: 't2.L1.p',
      record: 'r',
      symbol: 's',
    });
    // Back/forward in the browser changes only the detail: the address stays.
    pasted.navigate(formatFragment({ sid: 'qwen', addr: 't2.L1.p' }));
    expect(other.getState()).toMatchObject({ addr: 't2.L1.p', record: null, symbol: null });
    // Pasting a link for the same address with a record writes the URL once, not twice.
    const before = pasted.writes.length;
    pasted.navigate(formatFragment({ sid: 'qwen', addr: 't3.L1.p', record: 'q' }));
    expect(other.getState()).toMatchObject({ addr: 't3.L1.p', record: 'q' });
    expect(pasted.writes.length).toBe(before);
  });
});

describe('inspector pointer follows the cursor', () => {
  const pointer = {
    digest: 'd1',
    addr: null,
    spec: null,
    via: [],
    status: 'frontier',
    reason: null,
    shape: null,
    dtype: null,
  } as PointerRef;
  it('clears the chosen pointer on every cursor change', () => {
    useCursor.getState().setSession('tiny', 't0.L0.h');
    useUi.getState().set({ inspect: pointer });
    useCursor.getState().go('t1.L0.h');
    expect(useUi.getState().inspect).toBeNull();
    useUi.getState().set({ inspect: pointer });
    useCursor.getState().back();
    expect(useUi.getState().inspect).toBeNull();
    useUi.getState().set({ inspect: pointer });
    useCursor.getState().forward();
    expect(useUi.getState().inspect).toBeNull();
    useUi.getState().set({ inspect: pointer });
    useCursor.getState().setSession('qwen', 't2.L1.p');
    expect(useUi.getState().inspect).toBeNull();
    // Same address: nothing moved, so the choice stands.
    useUi.getState().set({ inspect: pointer });
    useCursor.getState().go('t2.L1.p');
    expect(useUi.getState().inspect).toBe(pointer);
  });
});
