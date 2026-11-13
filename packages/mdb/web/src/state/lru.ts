// Bounded caches. Immutable (a new Map per write) so zustand sees a change; sizes are small and
// fixed, so the copy is cheap. Recency = insertion order; a write or `touch` moves a key to the end.

export function lruSet<V>(
  map: ReadonlyMap<string, V>,
  key: string,
  value: V,
  max: number,
): Map<string, V> {
  const next = new Map(map);
  next.delete(key);
  next.set(key, value);
  while (next.size > max) {
    const oldest = next.keys().next();
    if (oldest.done) break;
    next.delete(oldest.value);
  }
  return next;
}
