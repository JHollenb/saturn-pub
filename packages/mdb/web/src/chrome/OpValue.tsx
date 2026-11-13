// One labelled value of an op result, drawn through NumView: a number, a summary (`max_abs …
// norm …`), or the unresolved/refused state with its reason. See `labelled.ts`.
import { NumView } from './NumView';
import { valOf } from './labelled';

export function OpValue({
  value,
  what,
  showReason = true,
  compact = false,
}: {
  value: unknown;
  /** Field name used in the reason when the value is missing. */
  what?: string;
  showReason?: boolean;
  compact?: boolean;
}) {
  const val = valOf(value, what);
  if (val.kind === 'num') return <NumView n={val.n} compact={compact} showReason={showReason} />;
  if (val.kind === 'absent') return <NumView n={val.n} showReason={showReason} />;
  return (
    <span className="op-summary">
      {val.parts.map(([key, n]) => (
        <span key={key}>
          <span className="faint">{key} </span>
          <NumView n={n} compact={compact} />{' '}
        </span>
      ))}
    </span>
  );
}
