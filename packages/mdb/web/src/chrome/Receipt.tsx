// A native receipt id (the record that a transition really ran). Click copies the full id.
export function Receipt({ id }: { id: string }) {
  return (
    <button
      type="button"
      className="chip mono"
      title={`receipt ${id} (click to copy)`}
      onClick={() => void navigator.clipboard?.writeText(id).catch(() => undefined)}
    >
      receipt {id.slice(0, 10)}
    </button>
  );
}
