// Shared UI settings (never a private selection: the cursor owns "where", and which record or
// symbol is open lives in the cursor store because it is part of the URL).
import { create } from 'zustand';
import type { PointerRef } from '../api/types.gen';
import { useCursor } from './cursor';

export type Theme = 'system' | 'light' | 'dark';

interface UiState {
  lens: string;
  zoom: 'slice' | 'session';
  drawerOpen: boolean;
  recordKind: string | null;
  symbolsAtCursor: boolean;
  /** Pointer chosen for reduction. Valid only for the address it was chosen at: cleared on every
   *  cursor move (see the subscription below). */
  inspect: PointerRef | null;
  help: boolean;
  command: boolean;
  theme: Theme;
  set(patch: Partial<Omit<UiState, 'set'>>): void;
}

export const useUi = create<UiState>((set) => ({
  lens: 'cuts',
  zoom: 'slice',
  drawerOpen: true,
  recordKind: null,
  symbolsAtCursor: false,
  inspect: null,
  help: false,
  command: false,
  theme: 'system',
  set: (patch) => set(patch),
}));

/** A chosen pointer belongs to the address it was chosen at; any cursor change (go, back,
 *  forward, URL edit, session switch) drops it so the inspector never reduces a pointer of a
 *  different address than the cursor. Callers that move the cursor and choose a pointer must move
 *  first, then choose. */
useCursor.subscribe((state, prev) => {
  if ((state.addr !== prev.addr || state.sid !== prev.sid) && useUi.getState().inspect) {
    useUi.setState({ inspect: null });
  }
});
