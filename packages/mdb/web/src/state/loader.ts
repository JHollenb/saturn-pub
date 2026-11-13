// Request pool: one network request per key however many views ask for it, and a request nobody
// listens to any more is aborted. A result is delivered only to subscribers that are still
// subscribed, so a stale request for a previous cursor can never overwrite the current view.
import type { Result } from '../api/client';

type Entry = {
  promise: Promise<Result<unknown>>;
  controller: AbortController;
  refs: number;
  settled: boolean;
};

export class RequestPool {
  private entries = new Map<string, Entry>();

  get size(): number {
    return this.entries.size;
  }

  /** Subscribe to `key`. `start` is called only by the first subscriber, with the abort signal.
   *  Returns the unsubscribe function; the last unsubscribe aborts an unfinished request. */
  subscribe(
    key: string,
    start: (signal: AbortSignal) => Promise<Result<unknown>>,
    onResult: (result: Result<unknown>) => void,
    onError: (error: unknown) => void,
  ): () => void {
    let entry = this.entries.get(key);
    if (!entry) {
      const controller = new AbortController();
      const created: Entry = {
        promise: start(controller.signal),
        controller,
        refs: 0,
        settled: false,
      };
      entry = created;
      this.entries.set(key, created);
      const done = () => {
        created.settled = true;
        if (this.entries.get(key) === created) this.entries.delete(key);
      };
      created.promise.then(done, done);
    }
    const joined = entry;
    joined.refs += 1;
    let live = true;
    joined.promise.then(
      (result) => {
        if (live) onResult(result);
      },
      (error: unknown) => {
        if (live) onError(error);
      },
    );
    return () => {
      if (!live) return;
      live = false;
      joined.refs -= 1;
      if (joined.refs === 0 && !joined.settled) {
        joined.controller.abort();
        if (this.entries.get(key) === joined) this.entries.delete(key);
      }
    };
  }
}
