// Predictions in flight: a what-if panel (the `whatif` op: sealed linear and exact predictions,
// the bound, the native measurement and the error split; the session is restored), a compare
// panel (the `compare` op: base, A, B, A+B, effects, interaction) and the run's forecasts (the
// `forecast` / `verify` ops). Nothing here changes the session's edits: "stage this edit" is a
// separate, explicit `set`. Pure: actions go through `act`.
import { useState } from 'react';
import type { LiveCheckpoint, LiveForecast } from '../../api/types.gen';
import { Count } from '../../chrome/NumView';
import { OpValue } from '../../chrome/OpValue';
import { at, isRec, str, strs, type Rec } from '../../chrome/labelled';
import type { Slot } from '../../state/inspect';
import { SlotState } from '../Live/SlotState';
import { SplitBar } from './SplitBar';

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
      <SlotState
        slot={whatif}
        what="whatif"
        onInterrupt={act.interrupt}
        idle="No what-if yet."
      />
      {result && <WhatIfResult result={result} busy={busy} act={act} />}
    </section>
  );
}

const OPS: readonly EditOp[] = ['replace', 'add', 'scale'];

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
  const address = str(edit.address);
  const opName = str(edit.op);
  const supplied = at(edit, 'value', 'value');
  const stageable =
    address !== null &&
    typeof supplied === 'number' &&
    opName !== null &&
    (OPS as readonly string[]).includes(opName);
  const exact = prediction.exact;
  return (
    <div className="whatif-result" data-testid="whatif-result">
      <div>
        edit <code>{address ?? '?'}</code> {opName ?? '?'}{' '}
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
              <OpValue value={at(prediction, 'linear', 'value') ?? prediction.linear} what="linear" />
              {str(at(prediction, 'linear', 'meaning')) && (
                <div className="faint">{str(at(prediction, 'linear', 'meaning'))}</div>
              )}
            </td>
          </tr>
          <tr>
            <th>sealed exact Δ</th>
            <td>
              <OpValue value={at(exact, 'value') ?? exact} what="exact" />{' '}
              <span className="faint">bound</span> <OpValue value={at(exact, 'bound')} what="bound" />
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
              <OpValue value={at(result, 'difference', 'native_minus_exact')} what="native minus exact" />{' '}
              <span className="faint">bound</span>{' '}
              <OpValue value={at(result, 'difference', 'native_minus_exact', 'bound')} what="bound" />
              {at(result, 'difference', 'native_minus_exact', 'within_bound') === true
                ? ' within'
                : at(result, 'difference', 'native_minus_exact', 'within_bound') === false
                  ? ' OUTSIDE'
                  : ''}
            </td>
          </tr>
        </tbody>
      </table>
      <SplitBar split={result.split} />
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
          disabled={busy || !stageable}
          onClick={() =>
            stageable && act.stage(address!.replace(/^t\d+\./, ''), supplied as number, opName as EditOp)
          }
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
      <SlotState slot={compare} what="compare" onInterrupt={act.interrupt} idle="No comparison yet." />
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
        <code>{str(at(result, 'sides', 'B')) ?? '?'}</code> → <code>{str(result.consumer) ?? '?'}</code>
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
                <th className="n">residual ≤ bound</th>
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
          <>
            <OpValue value={v.residual_max_abs} what="residual" compact /> ≤{' '}
            <OpValue value={v.bound_max ?? at(f.bound, 'max')} what="bound" compact />
          </>
        ) : (
          <>
            <span className="faint">bound</span>{' '}
            <OpValue value={at(f.bound, 'max')} what="bound" compact />
          </>
        )}
      </td>
      <td>
        <span className="chip">{str(v?.verdict) ?? f.status}</span>
        {str(v?.reason) && <span className="faint"> {str(v?.reason)}</span>}
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
