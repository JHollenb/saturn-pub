import { describe, expect, it } from 'vitest';
import { keyAction, type KeyLike } from './keys';

const frame = { prev: 't1.L0.h', next: 't3.L0.h' };
const el = (tagName: string, own = false) =>
  ({
    tagName,
    isContentEditable: false,
    closest: (s: string) => (own && s === '[data-own-keys]' ? {} : null),
  }) as unknown as EventTarget;
const press = (key: string, over: Partial<KeyLike> = {}): KeyLike => ({
  key,
  altKey: false,
  ctrlKey: false,
  metaKey: false,
  shiftKey: false,
  defaultPrevented: false,
  target: el('BODY'),
  ...over,
});

describe('keyboard handling', () => {
  it('walks recorded history with arrows on the page', () => {
    expect(keyAction(press('ArrowLeft'), frame)).toEqual({ kind: 'go', addr: 't1.L0.h' });
    expect(keyAction(press('ArrowDown'), frame)).toEqual({ kind: 'go', addr: 't3.L0.h' });
    expect(keyAction(press('ArrowLeft'), { prev: null, next: null })).toBeNull();
  });

  it('leaves arrows and letters to text inputs, selects and widgets that own their keys', () => {
    for (const tag of ['INPUT', 'TEXTAREA', 'SELECT']) {
      expect(keyAction(press('ArrowLeft', { target: el(tag) }), frame)).toBeNull();
      expect(keyAction(press('g', { target: el(tag) }), frame)).toBeNull();
      expect(keyAction(press('/', { target: el(tag) }), frame)).toBeNull();
    }
    expect(keyAction(press('ArrowRight', { target: el('CANVAS', true) }), frame)).toBeNull();
  });

  it('passes every Cmd/Ctrl combination through to the browser', () => {
    for (const key of ['ArrowLeft', 'ArrowRight', 'g', '/', '?', 'l', 'f', 'k']) {
      expect(keyAction(press(key, { metaKey: true }), frame)).toBeNull();
      expect(keyAction(press(key, { ctrlKey: true }), frame)).toBeNull();
    }
    expect(keyAction(press('ArrowLeft', { metaKey: true, altKey: true }), frame)).toBeNull();
  });

  it('does not act on an event another handler already used or an IME composition', () => {
    expect(keyAction(press('ArrowLeft', { defaultPrevented: true }), frame)).toBeNull();
    expect(keyAction(press('g', { isComposing: true }), frame)).toBeNull();
  });

  it('maps the remaining keys', () => {
    expect(keyAction(press('ArrowLeft', { altKey: true }), frame)).toEqual({ kind: 'back' });
    expect(keyAction(press('ArrowRight', { altKey: true }), frame)).toEqual({ kind: 'forward' });
    expect(keyAction(press('g'), frame)).toEqual({ kind: 'focus-address' });
    expect(keyAction(press('/'), frame)).toEqual({ kind: 'command' });
    expect(keyAction(press('?'), frame)).toEqual({ kind: 'help' });
    expect(keyAction(press('Escape', { target: el('INPUT') }), frame)).toEqual({ kind: 'close' });
    expect(keyAction(press('x'), frame)).toBeNull();
  });
});
