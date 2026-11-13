"""Reusable observation-only native FLUX joint ports for component assays.

This observes original module/dispatcher values. It does not replace arithmetic,
provide an ATen trace or change the declared component-state closure.
"""


class NativeJointPorts:
    def __init__(self, block, native, counts, image_indices, text_indices, *, image_rows=1024):
        import torch
        from saturn_pub.values import describe
        self.describe = describe
        self.native, self.counts = native, counts
        self.ix, self.txix = image_indices, text_indices
        self.image_rows = image_rows
        self.original = native.dispatch_attention_fn
        self.fields, self.ports, self.handles = None, None, []
        def count(key):
            def hook(module, args):
                counts[key] += 1
            return hook
        self.handles += [block.register_forward_pre_hook(count("native_blocks")),
            block.attn.register_forward_pre_hook(count("native_attention")),
            block.ff.register_forward_pre_hook(count("native_image_ff")),
            block.ff_context.register_forward_pre_hook(count("native_text_ff"))]
        self.handles += [m.register_forward_pre_hook(count("native_linears")) for m in block.modules() if isinstance(m, torch.nn.Linear)]
        def attn_input(module, args, kwargs):
            self.observe("image.norm", kwargs["hidden_states"][:, self.ix], "attn:input.hidden_states[image-witness]")
            self.observe("text.norm", kwargs["encoder_hidden_states"][:, self.txix], "attn:input.encoder_hidden_states[text-witness]")
        self.handles.append(block.attn.register_forward_pre_hook(attn_input, with_kwargs=True))
        def returning(field, selector, port):
            def hook(module, args, output):
                self.observe(field, selector(output), port)
                return output
            return hook
        for stream, ff, selected in (("image", block.ff, self.ix), ("text", block.ff_context, self.txix)):
            for field, module in (("projected", ff.linear_in), ("activated", ff.act_fn.gate_fn), ("product", ff.act_fn), ("mlp", ff)):
                port = next(n for n,m in block.named_modules() if m is module)
                self.handles.append(module.register_forward_hook(returning(stream+"."+field, lambda v, selected=selected:v[:,selected], port+":return[witness]")))
            def ff_input(module, args, stream=stream, selected=selected):
                self.observe(stream+".mlp_input", args[0][:,selected], ("ff" if stream=="image" else "ff_context")+":input[witness]")
            self.handles.append(ff.register_forward_pre_hook(ff_input))
        self.handles.append(block.attn.register_forward_hook(returning("image.attention",lambda v:v[0][:,self.ix],"attn:return.image[witness]")))
        self.handles.append(block.attn.register_forward_hook(returning("text.attention",lambda v:v[1][:,self.txix],"attn:return.text[witness]")))

    def observe(self, field, value, port):
        if self.fields is None or field in self.fields:
            raise ValueError("unscoped or duplicate native port: "+field)
        self.fields[field] = value.detach().float().cpu().numpy().copy()
        self.ports[field] = dict(port=port,descriptor=self.describe(value))

    def dispatch(self, q, k, v, *args, **kwargs):
        self.counts["native_sdpa"] += 1
        nt = k.shape[1]-self.image_rows
        for stream, positions in (("image",[nt+i for i in self.ix]),("text",self.txix)):
            for field, value in (("q",q),("k",k),("v",v)):
                self.observe(stream+"."+field,value[:,positions].permute(0,2,1,3),"native.dispatch_attention_fn:input."+field+"["+stream+"-witness]")
        result=self.original(q,k,v,*args,**kwargs)
        for stream, positions in (("image",[nt+i for i in self.ix]),("text",self.txix)):
            self.observe(stream+".head",result[:,positions].permute(0,2,1,3),"native.dispatch_attention_fn:return["+stream+"-witness]")
        return result

    def begin(self):
        if self.fields is not None:
            raise ValueError("native observation already active")
        self.fields,self.ports={},{}

    def finish(self, payload):
        for stream in ("image","text"):
            self.observe(stream,payload[stream],"native.block:return."+stream+"[complete]")
        fields,ports=self.fields,self.ports
        self.fields,self.ports=None,None
        return fields,ports

    def __enter__(self):
        if self.native.dispatch_attention_fn is not self.original:
            raise ValueError("native dispatcher already modified")
        self.native.dispatch_attention_fn=self.dispatch
        return self

    def __exit__(self, *error):
        self.native.dispatch_attention_fn=self.original
        for h in self.handles:h.remove()
        return False
