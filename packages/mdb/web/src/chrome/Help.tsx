// `?` overlay: keys and the evidence legend.
import { EV_CLASSES } from './evidence';
import { EvidenceMark } from './EvidenceMark';
import { useUi } from '../state/ui';

const KEYS: [string, string][] = [
  ['← / ↑', 'previous recorded state on this branch'],
  ['→ / ↓', 'next recorded state on this branch'],
  ['Alt+← / Alt+→', 'cursor history back / forward'],
  ['g', 'go to address (focus the address bar)'],
  ['/', 'command line (navigation only)'],
  ['?', 'this help'],
  ['Esc', 'close overlays'],
];

const EV_TEXT: Record<string, string> = {
  measured: 'read from retained bytes or a native receipt',
  'derived-exact': 'computed exactly from measured values (chip names the proof)',
  'derived-local': 'local approximation, e.g. a linearisation (chip names the proof)',
  bound: 'a proven bound, not a value',
  supplied: 'given by a person or a declaration, not measured',
  unresolved: 'the project does not hold this value; shown empty, never zero',
  refused: 'the server or renderer refused to show it',
};

export function Help() {
  const open = useUi((s) => s.help);
  const set = useUi((s) => s.set);
  if (!open) return null;
  return (
    <div className="overlay" role="dialog" aria-label="help" onClick={() => set({ help: false })}>
      <div
        className="panel panel-body"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 560 }}
      >
        <h3 style={{ marginTop: 0 }}>Keys</h3>
        <table>
          <tbody>
            {KEYS.map(([k, v]) => (
              <tr key={k}>
                <td className="mono">{k}</td>
                <td>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h3>Evidence</h3>
        <table>
          <tbody>
            {EV_CLASSES.map((ev) => (
              <tr key={ev}>
                <td>
                  <EvidenceMark ev={ev} /> {ev}
                </td>
                <td>{EV_TEXT[ev]}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted">Colour marks roles only. Evidence is shape and fill.</p>
      </div>
    </div>
  );
}
