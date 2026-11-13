// NumView is the ONLY component that prints a measured/derived/supplied number. It refuses a
// number without an evidence class: in development it throws; in production it renders a
// refusal. Unresolved values are an empty dashed box with the reason, never zero.
// `Count` prints structural integers (ids, indices, shapes, counts), which carry no class.
import type { Num } from '../api/types.gen';
import { EvidenceMark } from './EvidenceMark';
import { formatValue, isEv, needsChip, numNote } from './evidence';

export class EvidenceMissing extends Error {}

const DEV = import.meta.env.DEV;

interface Props {
  n: Num | null | undefined;
  /** Show the unresolved/refused reason as text (not only as a tooltip). */
  showReason?: boolean;
  /** Hide the mark (for dense tables that show the class elsewhere). Never hides the chip. */
  bare?: boolean;
  /** Chip carries the class only; the proof text goes in the tooltip (dense tables). */
  compact?: boolean;
}

export function NumView({ n, showReason = false, bare = false, compact = false }: Props) {
  if (!n || !isEv((n as Partial<Num>).ev)) {
    if (DEV)
      throw new EvidenceMissing(
        `NumView got a number without evidence class: ${JSON.stringify(n)}`,
      );
    return (
      <span className="num-refused" data-ev="refused" title="number has no evidence class">
        refused: no evidence class
      </span>
    );
  }
  if (n.ev === 'unresolved') {
    const reason = n.reason ?? 'unresolved';
    return (
      <span data-ev="unresolved" title={reason}>
        <span className="num-unresolved" aria-label={`unresolved: ${reason}`}>
          {showReason ? reason : ' '}
        </span>
      </span>
    );
  }
  if (n.ev === 'refused' || n.v === null || n.v === undefined) {
    const reason = n.reason ?? 'refused';
    return (
      <span className="num-refused" data-ev="refused" title={reason}>
        refused{showReason ? `: ${reason}` : ''}
      </span>
    );
  }
  const text = formatValue(n.v);
  const note = numNote(n);
  const title = [
    n.ev,
    n.unit,
    n.proof && `proof ${n.proof}`,
    n.ptr && `from ${n.ptr}`,
    n.recorded_authority && `recorded as ${n.recorded_authority}`,
    note,
    `value ${String(n.v)}`,
  ]
    .filter(Boolean)
    .join(' · ');
  return (
    <span className="num" data-ev={n.ev} title={title}>
      {!bare && <EvidenceMark ev={n.ev} proof={n.proof} />} {text}
      {n.unit ? <span className="faint"> {n.unit}</span> : null}
      {showReason && note && <span className="faint"> {note}</span>}
      {needsChip(n.ev) && (
        <span className="ev-chip">
          {n.ev}
          {n.proof && !compact ? ` · ${n.proof}` : ''}
        </span>
      )}
    </span>
  );
}

/** A structural integer (count, index, page number). Not evidence, so no mark. */
export function Count({ n, unit }: { n: number; unit?: string }) {
  return (
    <span className="num" data-structural="">
      {String(Math.trunc(n))}
      {unit ? ` ${unit}` : ''}
    </span>
  );
}
