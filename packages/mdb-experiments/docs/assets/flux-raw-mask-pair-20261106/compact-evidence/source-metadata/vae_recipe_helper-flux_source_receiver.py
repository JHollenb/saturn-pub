"""Source-defined text response across two saved FLUX receivers.

Pure mathematical suffix, source recipes and observation-only single-block ports.
The single-block MLP and attention are parallel; joint blocks are sequential.
"""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path


def helper(name):
    return runpy.run_path(str(Path(__file__).with_name(name)))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    from collections.abc import Mapping
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def immutable_mapping(value):
        if isinstance(value,Mapping):return dict(value)
        raise TypeError('unsupported JSON value: '+type(value).__name__)
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False,default=immutable_mapping)
        f.write('\n')


def arms():
    return [dict(id=name, receiver=receiver, kind=kind) for name, receiver, kind in (
        ('left.base', 'c0-o0', 'base'), ('left.delta', 'c0-o0', 'delta'),
        ('right.base', 'c0-o1', 'base'), ('right.delta', 'c0-o1', 'delta'),
        ('right.donor', 'c0-o1', 'donor'), ('right.permute', 'c0-o1', 'permute'),
        ('right.sham', 'c0-o1', 'sham'))]


def text_input(texts, arm):
    import numpy as np
    bf16 = helper('flux_reentry_reference.py')['bf16']
    receiver = texts[arm['receiver']]
    delta = texts['c1-o0'].astype(np.float32) - texts['c0-o0'].astype(np.float32)
    if arm['kind'] in ('base', 'sham'):
        return np.array(receiver, copy=True)
    if arm['kind'] == 'donor':
        return np.array(texts['c1-o1'], copy=True)
    if arm['kind'] == 'permute':
        delta = np.roll(delta, 1, axis=1)
    return bf16(receiver.astype(np.float32) + delta)


class RecipeBank:
    """Read validated original BF16 payloads through a bounded immutable cache.

    No exported parameter files, native operator call or response fitting.
    Decoded FP32 block-local arrays are temporary.
    """
    def __init__(self, manifest, root, max_cache_bytes=256 << 20):
        from collections import OrderedDict

        from mdb import parameter_artifacts as api
        manifest = api.validate_manifest(manifest)
        self.bits, self.records, self.bytes, self.reads = OrderedDict(), {}, 0, 0
        self.root, self.limit = Path(root), max_cache_bytes
        self.signatures, self.offsets = {}, {}
        self.retained = self.peak_retained = self.payload_reads = self.cache_hits = 0
        if type(max_cache_bytes) is not int or max_cache_bytes < 1:
            raise ValueError('finite positive parameter-cache bound required')
        # Original public retriever verifies each complete source file once and
        # every named header/offset. Hash each payload when first loaded, and
        # reject file-signature drift on every read, including cache hits.
        for entry, stream, size in api._source_entries(manifest, root):
            source = entry['source']
            self.records[entry['path']] = entry
            self.offsets[entry['path']] = stream.tell(), size
            path=self.root/source['file']
            stat=path.stat()
            self.signatures[source['file']]=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
            self.bytes += size

    def load(self, record):
        import base64

        import numpy as np
        if record is None:
            return None
        entry = self.records.get(record['path'])
        if (entry is None or record['sha256'] != entry['sha256']
                or record['descriptor']['sha256'] != entry['source']['payload_sha256']
                or record['parameter_tensor'] not in [entry['source']['tensor'], *entry.get('checkpoint_tensor_aliases', [])]):
            raise ValueError('parameter outside original named recipe')
        path=self.root/entry['source']['file'];stat=path.stat()
        signature=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
        if signature!=self.signatures[entry['source']['file']]:
            raise ValueError('original checkpoint changed during forecast')
        key=record['path']
        if key in self.bits:
            self.cache_hits+=1;self.bits.move_to_end(key);raw=self.bits[key]
        else:
            offset,size=self.offsets[key]
            with path.open('rb') as stream:
                stream.seek(offset);raw=stream.read(size)
            if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=entry['source']['payload_sha256']:
                raise ValueError('original parameter payload mismatch')
            if hashlib.sha256(base64.b64decode(entry['npy_prefix_b64'])+raw).hexdigest()!=entry['sha256']:
                raise ValueError('original parameter archive identity mismatch')
            self.payload_reads+=1
            while self.bits and self.retained+size>self.limit:
                _,old=self.bits.popitem(last=False);self.retained-=len(old)
            if size<=self.limit:
                self.bits[key]=raw;self.retained+=size
                self.peak_retained=max(self.peak_retained,self.retained)
        self.reads += 1
        return (np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16).view(np.float32).reshape(record['stored_shape'])


def suffix_events(payload, binding, load, bank, config):
    """Fifty own-state transitions from observed step2/single0 to terminal."""
    import numpy as np
    tr = helper('flux_transition_reference.py')
    re = helper('flux_reentry_reference.py')
    bo = helper('flux_step_boundary_reference.py')
    latent = payload['latent']
    image, text, context = payload['image'], payload['text'], payload['context']
    for logical in (2, 3):
        schedule = binding['pass_schedules'][str(logical)]
        if logical == 3:
            spec = binding['project']
            r = re['predict_project'](latent, payload['conditioning'], payload['img_ids'], payload['txt_ids'],
                load(schedule['timestep']), {k:bank.load(v) for k,v in spec['parameters'].items()},
                axes_dims_rope=spec['axes_dims_rope'], rope_theta=spec['rope_theta'],
                timestep_channels=spec['timestep_channels'], projection_chunk=config['projection_chunk'])
            image, text, context = r['image'], r['text'], r['context']
            yield logical, 0, dict(image=image, text=text), {}
            for index in range(5):
                spec = binding['joint'+str(index)]
                r = re['predict_joint'](image, text, context, {k:bank.load(v) for k,v in spec['parameters'].items()},
                    heads=spec['heads'], layernorm_eps=spec['layernorm_eps'], qk_eps=spec['qk_eps'],
                    query_chunk=config['query_chunk'], projection_chunk=config['projection_chunk'], retain_fields=False)
                image, text = r['image'], r['text']
                yield logical, index+1, dict(image=image, text=text), {}
        current = np.concatenate((text, image), axis=1)
        for index in range(20):
            spec = binding['final_blocks'][str(index)]
            r = tr['predict_transition'](current, context['single_mod'], **{k:bank.load(v) for k,v in spec['parameters'].items()},
                heads=spec['heads'], layernorm_eps=spec['layernorm_eps'], qk_eps=spec['qk_eps'],
                rope_cos=context['rope'][0], rope_sin=context['rope'][1], query_chunk=config['query_chunk'],
                projection_chunk=config['projection_chunk'], neuron_chunk=config['neuron_chunk'],
                retain_fields=False, text_rows=None, sample_indices=config['sample_indices'])
            current = r['output']
            text, image = current[:, :binding['text_rows']], current[:, binding['text_rows']:]
            ports = {k:v for k,v in r['sampled'].items()
                if hasattr(v, 'dtype') and np.issubdtype(v.dtype,np.floating)} if logical == 2 and index < 2 else {}
            yield logical, index+6, dict(image=image, text=text), ports
        spec = binding['readout']
        r = bo['predict_readout'](image, context['temb'], **{k:bank.load(v) for k,v in spec['parameters'].items()},
            eps=spec['eps'], neuron_chunk=config['projection_chunk'], sample_indices=config['image_sample_indices'], retain_fields=False)
        yield logical, 26, dict(noise=r['noise']), {}
        r = bo['predict_flow_commit'](latent, r['noise'], load(schedule['sigma']), load(schedule['sigma_next']))
        latent = r['latent']
        yield logical, 27, dict(latent=latent), {}


def retained(logical, local):
    return (logical == 2 and local in (6, 7, 27)) or (logical == 3 and local in (0, 1, 27))


def array(out, value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    value = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(value).all() or np.any(value.view(np.uint32) & 65535):
        raise ValueError('expected exact finite BF16 widening')
    bits = (value.view(np.uint32) >> 16).astype('<u2')
    name = hashlib.sha256(json.dumps(list(value.shape)).encode()+bits.tobytes()).hexdigest()+'.npy'
    path = Path(out)/'arrays'/name
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with path.open('xb') as f:
            np.save(f, bits, allow_pickle=False)
    return dict(path='arrays/'+name, sha256=sha(path), shape=list(value.shape), encoding='bf16-bits-v2')


def load_array(root, record):
    import numpy as np
    path = Path(root)/record['path']
    if sha(path) != record['sha256'] or record['encoding'] not in ('bf16-bits-v1','bf16-bits-v2'):
        raise ValueError('array custody mismatch')
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.dtype('<u2'):
        raise ValueError('array geometry mismatch')
    if tuple(value.shape)!=tuple(record['shape']):
        # Legacy v1 addressed only the raw bit sequence, so two views could
        # share one NPY header. The sealed record declares its logical view.
        # Preserve every bit and permit this view only in that legacy format.
        if record['encoding']!='bf16-bits-v1' or value.size!=np.prod(record['shape']):
            raise ValueError('array geometry mismatch')
        value=value.reshape(record['shape'])
    return (value.astype(np.uint32) << 16).view(np.float32)


class SinglePorts:
    """Observe original single-block parallel branches; never replace returns."""
    def __init__(self, block, native, indices):
        self.block, self.native, self.ix = block, native, indices
        self.original = native.dispatch_attention_fn
        self.fields, self.handles = {}, []

    def note(self, name, value):
        if name in self.fields:
            raise ValueError('duplicate native single port')
        self.fields[name] = value.detach().float().cpu().numpy().copy()

    def dispatch(self, q, k, v, *args, **kwargs):
        for name, value in (('q_rope', q), ('k_rope', k), ('v', v)):
            self.note(name, value[:, self.ix].flatten(2, 3))
        result = self.original(q, k, v, *args, **kwargs)
        self.note('attention', result[:, self.ix].flatten(2, 3))
        return result

    def __enter__(self):
        width = self.block.attn.inner_dim
        def returning(name, select):
            def hook(module, args, value):
                self.note(name, select(value))
            return hook
        self.handles = [
            self.block.norm.register_forward_hook(returning('layernorm', lambda v:v[:, self.ix])),
            self.block.attn.to_qkv_mlp_proj.register_forward_pre_hook(lambda m,a:self.note('modulated',a[0][:,self.ix])),
            self.block.attn.to_qkv_mlp_proj.register_forward_hook(returning('q_raw', lambda v:v[:, self.ix, :width])),
            self.block.attn.to_qkv_mlp_proj.register_forward_hook(returning('k_raw', lambda v:v[:, self.ix, width:2*width])),
            self.block.attn.norm_q.register_forward_hook(returning('q_norm', lambda v:v[:, self.ix].flatten(2,3))),
            self.block.attn.norm_k.register_forward_hook(returning('k_norm', lambda v:v[:, self.ix].flatten(2,3))),
            self.block.attn.to_out.register_forward_pre_hook(lambda m,a:self.note('mlp_full',a[0][:,self.ix,width:])),
            self.block.attn.to_out.register_forward_hook(returning('projection', lambda v:v[:,self.ix]))]
        self.native.dispatch_attention_fn = self.dispatch
        return self

    def __exit__(self, *args):
        self.native.dispatch_attention_fn = self.original
        for handle in self.handles:
            handle.remove()
