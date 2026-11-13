// Write static/manifest.json: a sha256 over the web sources, so a built bundle names its inputs.
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const web = fileURLToPath(new URL('..', import.meta.url));
const out = join(web, '..', 'src', 'mdb', 'serve', 'static');
const roots = ['src', 'index.html', 'package.json', 'package-lock.json', 'vite.config.ts', 'tsconfig.json'];

function walk(path) {
  if (statSync(path).isDirectory()) return readdirSync(path).sort().flatMap((n) => walk(join(path, n)));
  return [path];
}

const files = roots.flatMap((r) => walk(join(web, r))).map((p) => relative(web, p)).sort();
const hash = createHash('sha256');
for (const file of files) {
  hash.update(file + '\0');
  hash.update(readFileSync(join(web, file)));
  hash.update('\0');
}
const manifest = { schema: 'mdb-web-manifest-v1', sourceHash: hash.digest('hex'), files: files.length };
writeFileSync(join(out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(`manifest: ${files.length} source files, sha256 ${manifest.sourceHash.slice(0, 16)}`);
