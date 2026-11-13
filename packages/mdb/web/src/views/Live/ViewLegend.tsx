// One compact legend line per live view: what each evidence mark means, and the "show labels"
// toggle that spells the authority out beside every number again. Unresolved and refused values
// always spell themselves out with their reason, labels or not.
import { EvidenceMark } from '../../chrome/EvidenceMark';

export function ViewLegend({
  labels,
  onLabels,
}: {
  labels: boolean;
  onLabels: (on: boolean) => void;
}) {
  return (
    <div className="view-legend faint" data-testid="view-legend">
      <span>
        <EvidenceMark ev="measured" /> measured
      </span>
      <span>
        <EvidenceMark ev="derived-exact" /> derived or bound
      </span>
      <span>
        <EvidenceMark ev="supplied" /> supplied
      </span>
      <span>
        <EvidenceMark ev="unresolved" /> unresolved
      </span>
      <span>
        <EvidenceMark ev="refused" /> refused
      </span>
      <span>· hover a value for its authority, proof and full value</span>
      <label className="view-legend-toggle">
        <input
          type="checkbox"
          checked={labels}
          onChange={(e) => onLabels(e.target.checked)}
        />{' '}
        show labels
      </label>
    </div>
  );
}
