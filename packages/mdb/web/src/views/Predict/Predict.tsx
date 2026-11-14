// Predictions in flight: a what-if panel (the `whatif` op: sealed linear, exact and emulated
// predictions, the exact chain's worst-case interval or the note saying where it stopped being
// informative, the native measurement, the verdict against the emulated prediction, the error
// split and each run's per-stage containment; the session is restored), a compare
// panel (the `compare` op: base, A, B, A+B, effects, interaction) and the run's forecasts (the
// `forecast` / `verify` ops). Nothing here changes the session's edits: "stage this edit" is a
// separate, explicit `set`. Pure: actions go through `act`.
import { useState } from 'react';
import type { LiveCheckpoint, LiveForecast } from '../../api/types.gen';
import { Count } from '../../chrome/NumView';
import { OpValue } from '../../chrome/OpValue';
import { at, editAddress, isRec, str, strs, type Rec } from '../../chrome/labelled';
import type { Slot } from '../../state/inspect';
import { SlotState } from '../Live/SlotState';
import { SplitBar } from './SplitBar';
import { ResidualVsFloor, VerdictChip } from './VerifyVerdict';

export type EditOp = 'replace' | 'add' | 'scale';

export interface PredictActions {
  whatif(target: string, value: number, op: EditOp, consumer: string): void;
  /** Stage the edit (the `set` op): the only action here that changes the session's edits. */
  stage(target: string, value: number, op: EditOp): void;
  compare(a: string, b: string, consumer: string | null): void;
  forecast(target: string | null): void;
  verify(ids: number[] | null): void;
  interrupt(): void;
}

export interface PredictProps {
  whatif: Slot;
  compare: Slot;
  forecast: Slot;
  forecasts: LiveForecast[];
  checkpoints: LiveCheckpoint[];
  /** The port under the cursor (`L1.m`), the what-if target's default. */
  port: string | null;
  where: string | null;
  busy: boolean;
  act: PredictActions;
}

export function Predict(props: PredictProps) {
  return (
    <div className="predict" data-testid="predict">
      <WhatIf {...props} />
      <Compare {...props} />
      <Forecasts {...props} />
    </div>
  );
}

// --------------------------------------------------------------------------- what-if

function WhatIf({ whatif, port, busy, act }: PredictProps) {
  const [target, setTarget] = useState('');
  const [op, setOp] = useState<EditOp>('add');
  const [value, setValue] = useState('');
  const [consumer, setConsumer] = useState('');
  const chosen = target.trim() || port || '';
  const number = Number(value);
  const valid = value.trim() !== '' && Number.isFinite(number);
  const ready = valid && !!chosen && !!consumer.trim();
  const result = whatif.state === 'done' ? whatif.result : null;
  return (
    <section className="predict-panel" aria-label="what if">
      <h3>What if</h3>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (ready) act.whatif(chosen, number, op, consumer.trim());
        }}
      >
        <input
          className="mono"
          aria-label="whatif target"
          placeholder={port ?? 'L1.m'}
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          size={10}
        />
        <select aria-label="whatif op" value={op} onChange={(e) => setOp(e.target.value as EditOp)}>
          <option value="replace">replace</option>
          <option value="add">add</option>
          <option value="scale">scale</option>
        </select>
        <input
          className="mono"
          aria-label="whatif value"
          inputMode="decimal"
          placeholder="value"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          size={6}
        />
        <span className="muted">→</span>
        <input
          className="mono"
          aria-label="whatif consumer"
          placeholder="M[4,9] | logits[7] | L1.h_out"
          value={consumer}
          onChange={(e) => setConsumer(e.target.value)}
          size={14}
        />
        <button type="submit" disabled={busy || !ready}>
          predict and measure
        </button>
      </form>
      <div className="faint">
        Runs <code>whatif</code>: seals the prediction, then measures natively and restores the
        session. It does not stage the edit.
      </div>
      <SlotState slot={whatif} what="whatif" onInterrupt={act.interrupt} idle="No what-if yet." />
      {result && <WhatIfResult result={result} busy={busy} act={act} />}
    </section>
  );
}

const OPS: readonly EditOp[] = ['replace', 'add', 'scale'];

/** The full address of a what-if's edit: the port with its selector (`t17.L22.A[h6]`), since
 *  the selector limits what the edit wrote. Null when the result names no address. */
export function editAddressOf(result: Rec): string | null {
  const edit = isRec(result.edit) ? result.edit : {};
  const address = str(edit.address);
  return address === null ? null : editAddress(address, edit.selection);
}

/** The `set` that stages a what-if's edit as it was measured (same port, selector, op and
 *  value), or null when the result does not say enough to stage it. */
export function stageOf(result: Rec): { target: string; value: number; op: EditOp } | null {
  const edit = isRec(result.edit) ? result.edit : {};
  const address = editAddressOf(result);
  const op = str(edit.op);
  const value = at(edit, 'value', 'value');
  if (address === null || typeof value !== 'number' || op === null) return null;
  if (!(OPS as readonly string[]).includes(op)) return null;
  return { target: address.replace(/^t\d+\./, ''), value, op: op as EditOp };
}

export function WhatIfResult({
  result,
  busy,
  act,
}: {
  result: Rec;
  busy: boolean;
  act: PredictActions;
}) {
  const edit = isRec(result.edit) ? result.edit : {};
  const prediction = isRec(result.prediction) ? result.prediction : {};
  const measured = isRec(result.measured) ? result.measured : {};
  const session = isRec(result.session) ? result.session : {};
  const moves = strs(measured.moves);
  const address = editAddressOf(result);
  const opName = str(edit.op);
  const staged = stageOf(result);
  const exact = prediction.exact;
  const emulated = prediction.emulated;
  const effect = isRec(result.effect) ? result.effect : {};
  const otherEdits = strs(at(prediction, 'other_edits', 'applied'));
  return (
    <div className="whatif-result" data-testid="whatif-result">
      <div>
        edit <code data-testid="whatif-edit-address">{address ?? '?'}</code> {opName ?? '?'}{' '}
        <OpValue value={edit.value} what="edit value" /> → consumer{' '}
        <code>{str(at(result, 'consumer', 'address')) ?? '?'}</code>{' '}
        <span className="faint">
          (mode {str(result.mode) ?? '?'}
          {str(result.record) ? `, record ${str(result.record)!.slice(0, 16)}` : ''})
        </span>
      </div>
      <table className="whatif-table">
        <tbody>
          <tr>
            <th>sealed linear Δ</th>
            <td>
              <OpValue
                value={at(prediction, 'linear', 'value') ?? prediction.linear}
                what="linear"
              />
              {str(at(prediction, 'linear', 'meaning')) && (
                <div className="faint">{str(at(prediction, 'linear', 'meaning'))}</div>
              )}
            </td>
          </tr>
          <tr>
            <th>sealed exact Δ</th>
            <td>
              <OpValue value={at(exact, 'value') ?? exact} what="exact" />{' '}
              <BoundOrNote holder={exact} />
            </td>
          </tr>
          <tr>
            <th>sealed emulated Δ</th>
            <td>
              <OpValue value={at(emulated, 'value') ?? emulated} what="emulated" />
              {str(at(emulated, 'meaning')) && (
                <div className="faint">{str(at(emulated, 'meaning'))}</div>
              )}
            </td>
          </tr>
          <tr>
            <th>native Δ</th>
            <td>
              <OpValue value={measured.delta} what="measured delta" />{' '}
              <span className="faint">
                (base <OpValue value={measured.base} what="base" compact /> → edited{' '}
                <OpValue value={measured.edited} what="edited" compact />)
              </span>
            </td>
          </tr>
          <tr>
            <th>native − exact</th>
            <td>
              <OpValue
                value={at(result, 'difference', 'native_minus_exact')}
                what="native minus exact"
              />{' '}
              <BoundOrNote holder={at(result, 'difference', 'native_minus_exact')} />
            </td>
          </tr>
          <tr>
            <th>verdict</th>
            <td data-testid="whatif-verdict">
              <VerdictChip v={effect} status="?" />{' '}
              {effect.residual !== undefined && (
                <span className="faint">
                  residual <OpValue value={effect.residual} what="residual" compact />
                  {effect.floor !== undefined && (
                    <>
                      {' '}
                      vs floor <OpValue value={effect.floor} what="floor" compact />
                    </>
                  )}
                </span>
              )}
              {[str(effect.rests_on), str(effect.note)].filter(Boolean).map((line) => (
                <div key={line} className="faint">
                  {line}
                </div>
              ))}
            </td>
          </tr>
        </tbody>
      </table>
      <SplitBar split={result.split} />
      <StageLines stages={result.stages} />
      {otherEdits.length > 0 && (
        <div className="faint" data-testid="whatif-other-edits">
          other edits in the chains: {otherEdits.join(', ')}
          {str(at(prediction, 'other_edits', 'rule')) &&
            ` (${str(at(prediction, 'other_edits', 'rule'))})`}
        </div>
      )}
      <div className="whatif-session" data-testid="whatif-session">
        {session.unchanged === true ? (
          <>
            The session was restored: unchanged at <code>{str(result.where) ?? '?'}</code> after{' '}
            <Count n={moves.length} /> move{moves.length === 1 ? '' : 's'}
            {moves.length ? ` (last: ${moves[moves.length - 1]})` : ''}.
          </>
        ) : session.unchanged === false ? (
          <span className="num-refused">The op reports the session CHANGED.</span>
        ) : (
          <OpValue value={undefined} what="session restore report" />
        )}
      </div>
      <div className="whatif-stage">
        <button
          type="button"
          disabled={busy || !staged}
          onClick={() => staged && act.stage(staged.target, staged.value, staged.op)}
          title="the set op: stages this edit at the pending layer (commit applies it)"
        >
          stage this edit
        </button>{' '}
        <span className="faint">
          (<code>set</code>; the only button here that changes the session)
        </span>
      </div>
    </div>
  );
}

/** A chain's worst-case rounding interval, as information (never the verdict): the finite
 *  interval with whether the measured value lies within it, or the op's note saying where the
 *  interval stopped being informative and why (M56/M57: `bound_note`), or nothing. */
export function BoundOrNote({ holder }: { holder: unknown }) {
  if (!isRec(holder)) return null;
  if (holder.bound !== undefined) {
    return (
      <span title="worst-case rounding interval through the chain: information, not the verdict">
        <span className="faint">worst-case interval</span>{' '}
        <OpValue value={holder.bound} what="bound" />
        {holder.within_bound === true ? ' within' : holder.within_bound === false ? ' OUTSIDE' : ''}
      </span>
    );
  }
  const note = str(holder.bound_note);
  return note ? (
    <span className="faint" data-testid="bound-note">
      ({note})
    </span>
  ) : null;
}

/** Each run's per-stage containment on native operands (M56c): a per-stage bound, not a chain
 *  bound. `stages` is `{edited?, base?}` (a what-if) or one containment record (verify). */
export function StageLines({ stages }: { stages: unknown }) {
  if (!isRec(stages)) return null;
  const sides: [string, unknown][] =
    'text' in stages || 'state' in stages ? [['', stages]] : Object.entries(stages);
  return (
    <div className="stage-lines" data-testid="stage-lines">
      {sides.map(([side, s]) =>
        isRec(s) ? (
          <div key={side} className="faint" data-stages={side || 'one'}>
            {side ? `${side} run ` : ''}stages{' '}
            {s.state === 'refused' || s.state === 'unresolved' ? (
              <>
                {str(s.state)} ({str(s.reason) ?? '?'})
              </>
            ) : (
              <>
                [per-stage bound, not a chain bound]: {str(s.text) ?? '?'}
                {str(s.cost) && <span title="cost"> · {str(s.cost)}</span>}
              </>
            )}
          </div>
        ) : null,
      )}
    </div>
  );
}

// --------------------------------------------------------------------------- compare

const ARMS = ['base', 'A', 'B', 'A+B'] as const;

function Compare({ compare, checkpoints, busy, act }: PredictProps) {
  const [a, setA] = useState('');
  const [b, setB] = useState('');
  const [consumer, setConsumer] = useState('');
  const result = compare.state === 'done' ? compare.result : null;
  return (
    <section className="predict-panel" aria-label="compare">
      <h3>Compare two edits or checkpoints</h3>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (a.trim() && b.trim()) act.compare(a.trim(), b.trim(), consumer.trim() || null);
        }}
      >
        <input
          className="mono"
          aria-label="compare A"
          placeholder="A: L1.m=0.5 add | checkpoint"
          list="compare-checkpoints"
          value={a}
          onChange={(e) => setA(e.target.value)}
          size={16}
        />
        <input
          className="mono"
          aria-label="compare B"
          placeholder="B: L0.v=0.25 add"
          list="compare-checkpoints"
          value={b}
          onChange={(e) => setB(e.target.value)}
          size={16}
        />
        <input
          className="mono"
          aria-label="compare consumer"
          placeholder="consumer (default logits)"
          value={consumer}
          onChange={(e) => setConsumer(e.target.value)}
          size={12}
        />
        <datalist id="compare-checkpoints">
          {checkpoints.map((c) => (
            <option key={c.name} value={c.name} />
          ))}
        </datalist>
        <button type="submit" disabled={busy || !a.trim() || !b.trim()}>
          compare
        </button>
      </form>
      <SlotState
        slot={compare}
        what="compare"
        onInterrupt={act.interrupt}
        idle="No comparison yet."
      />
      {result && <CompareResult result={result} />}
    </section>
  );
}

export function CompareResult({ result }: { result: Rec }) {
  const measured = isRec(result.measured) ? result.measured : {};
  const effects = isRec(result.effects) ? result.effects : {};
  const interaction = isRec(result.interaction) ? result.interaction : undefined;
  const session = isRec(result.session) ? result.session : {};
  return (
    <div className="compare-result" data-testid="compare-result">
      <div>
        A <code>{str(at(result, 'sides', 'A')) ?? '?'}</code> · B{' '}
        <code>{str(at(result, 'sides', 'B')) ?? '?'}</code> →{' '}
        <code>{str(result.consumer) ?? '?'}</code>
      </div>
      <table>
        <thead>
          <tr>
            <th>arm</th>
            <th className="n">consumer (native)</th>
            <th className="n">effect vs base</th>
            <th className="n">moves</th>
          </tr>
        </thead>
        <tbody>
          {ARMS.map((arm) => {
            const m = measured[arm];
            const refusedArm = m === undefined && arm === 'A+B';
            return (
              <tr key={arm} data-arm={arm}>
                <td>{arm}</td>
                <td className="n">
                  {refusedArm ? (
                    <span className="faint">not run (see interaction)</span>
                  ) : (
                    <OpValue value={at(m, 'value') ?? m} what={`${arm} value`} />
                  )}
                </td>
                <td className="n">
                  {arm === 'base' ? (
                    <span className="faint">—</span>
                  ) : refusedArm && effects[arm] === undefined ? (
                    <span className="faint">not run</span>
                  ) : (
                    <OpValue value={effects[arm]} what={`${arm} effect`} />
                  )}
                </td>
                <td className="n faint" title={strs(at(m, 'moves')).join('; ')}>
                  {strs(at(m, 'moves')).length || ''}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div data-testid="compare-interaction">
        interaction{' '}
        {str(at(interaction, 'formula')) && (
          <span className="faint">{str(at(interaction, 'formula'))} = </span>
        )}
        <OpValue value={at(interaction, 'value') ?? interaction} what="interaction" />
      </div>
      <div className="faint">
        {session.unchanged === true
          ? `session unchanged at ${str(result.where) ?? '?'} (restored after each arm)`
          : session.unchanged === false
            ? 'the op reports the session CHANGED'
            : 'no session report'}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------- forecasts

function Forecasts({ forecast, forecasts, where, busy, act }: PredictProps) {
  const [target, setTarget] = useState('');
  const open = forecasts.filter((f) => f.status === 'open').length;
  return (
    <section className="predict-panel" aria-label="forecasts">
      <h3>Forecasts sealed on this run</h3>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          act.forecast(target.trim() || null);
          setTarget('');
        }}
      >
        <input
          className="mono"
          aria-label="forecast target"
          placeholder="next stop, ADDR or M[a,b]"
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          size={16}
        />
        <button type="submit" disabled={busy || where === null}>
          forecast
        </button>
        <button
          type="button"
          disabled={busy || open === 0}
          onClick={() => act.verify(null)}
          title="verify every open forecast whose target the cursor has passed"
        >
          verify reached ({open} open)
        </button>
      </form>
      <SlotState slot={forecast} what="forecast" onInterrupt={act.interrupt} idle="" />
      {forecasts.length ? (
        <div className="table-scroll">
          <table className="forecast-table">
            <thead>
              <tr>
                <th>id</th>
                <th>target</th>
                <th>sealed at</th>
                <th className="n">predicted</th>
                <th className="n">native</th>
                <th
                  className="n"
                  title="the verdict compares the measured residual with the floor (dtype spacings); the bound is information"
                >
                  residual vs floor
                </th>
                <th>verdict</th>
              </tr>
            </thead>
            <tbody>
              {forecasts.map((f) => (
                <ForecastRow key={f.id} f={f} busy={busy} act={act} />
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="faint">none sealed</div>
      )}
    </section>
  );
}

function ForecastRow({ f, busy, act }: { f: LiveForecast; busy: boolean; act: PredictActions }) {
  const v = f.verdict;
  return (
    <tr data-forecast={f.id} data-status={f.status}>
      <td>
        <Count n={f.id} />
      </td>
      <td className="mono">{f.target}</td>
      <td className="mono faint">{f.sealed_at}</td>
      <td className="n">
        <OpValue value={f.predicted} what="predicted" compact />
      </td>
      <td className="n">
        {v && v.native !== undefined ? (
          <OpValue value={v.native} what="native" compact />
        ) : (
          <span className="faint">not measured</span>
        )}
      </td>
      <td className="n">
        {v && v.residual_max_abs !== undefined ? (
          <ResidualVsFloor v={v} />
        ) : f.bound.max !== undefined ? (
          <>
            <span className="faint">bound</span>{' '}
            <OpValue value={at(f.bound, 'max')} what="bound" compact />
          </>
        ) : (
          <span className="faint" data-testid="forecast-bound-note">
            {str(f.bound.note) ??
              (str(f.bound.state)
                ? `worst-case interval ${str(f.bound.state)} (${str(f.bound.reason) ?? '?'})`
                : 'no worst-case interval')}
          </span>
        )}
      </td>
      <td>
        <VerdictChip v={v} status={f.status} />
        {f.status === 'open' && (
          <button type="button" disabled={busy} onClick={() => act.verify([f.id])}>
            verify
          </button>
        )}
      </td>
    </tr>
  );
}

/** The forecasts of a port address, latest first (for the pipeline's forecast column). */
export function forecastsOf(forecasts: LiveForecast[], addr: string): LiveForecast[] {
  return forecasts.filter((f) => f.target === addr).sort((x, y) => y.id - x.id);
}
