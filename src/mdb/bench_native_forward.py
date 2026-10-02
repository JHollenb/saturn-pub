"""Resumable execution of the installed native denoiser forward body.

The bytecode retains the native expressions and native block calls. Only Python
control flow is lowered to an explicit cursor. It does not recompute prior
blocks on restore. Source identity is bound into every frame. Unsupported
control flow fails closed rather than pretending hooks provide continuation.
"""
import ast
import inspect
import textwrap
from hashlib import sha256
import torch
from .bench_qwen import BenchError


class NativeForwardMachine:
    def __init__(self, module):
        self.module = module
        self.function = inspect.unwrap(type(module).forward)
        self.source = textwrap.dedent(inspect.getsource(self.function))
        self.source_sha256 = sha256(self.source.encode()).hexdigest()
        self.globals = self.function.__globals__
        self.nodes = []; self.code = []
        function = next(n for n in ast.parse(self.source).body if isinstance(n, ast.FunctionDef))
        for native in module.modules():
            if native._forward_hooks or native._forward_pre_hooks or "forward" in vars(native):
                raise BenchError("native block cuts require modules without forward hooks or overrides")
        for node in ast.walk(function):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                root = node
                while isinstance(root, ast.Attribute): root = root.value
                if isinstance(root, ast.Name) and root.id == "self":
                    raise BenchError("native forward mutates instance attributes outside the declared closure")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"setattr", "delattr"}:
                raise BenchError("native forward requires qualification of dynamic attribute mutation")
        self._lower(function.body)
        self.code.append(("return", ast.Constant(None)))
        self.modules = dict(module.named_modules())
        self.module_names = {id(m): n for n, m in self.modules.items()}
        self.boundary_count = sum(op == "boundary" for op, _ in self.code)
        if not self.boundary_count: raise BenchError("native forward has no qualified block sites")

    def _emit(self, op, arg): self.code.append((op, arg)); return len(self.code)-1

    def _lower(self, body, loop=None):
        for node in body:
            if isinstance(node, ast.If):
                test = self._emit("if", [node.test, None])
                self._lower(node.body, loop)
                jump = self._emit("jump", None)
                self.code[test][1][1] = len(self.code)
                self._lower(node.orelse, loop)
                self.code[jump] = ("jump", len(self.code))
            elif isinstance(node, ast.For):
                key = str(len(self.code)); self._emit("for_init", (key, node.iter))
                start = len(self.code); test = self._emit("for_next", [key, node.target, None])
                exits = []
                self._lower(node.body, (start, exits)); self._emit("jump", start)
                end = len(self.code); self.code[test][1][2] = end
                for pos in exits: self.code[pos] = ("jump", end)
                if node.orelse: raise BenchError("for-else native forward needs an explicit qualification")
            elif isinstance(node, ast.Break):
                if loop is None: raise BenchError("invalid native break")
                loop[1].append(self._emit("jump", None))
            elif isinstance(node, ast.Continue):
                if loop is None: raise BenchError("invalid native continue")
                self._emit("jump", loop[0])
            elif isinstance(node, ast.Return): self._emit("return", node.value or ast.Constant(None))
            else:
                if isinstance(node, (ast.With, ast.Try, ast.While, ast.AsyncFor)):
                    raise BenchError(f"unqualified native forward control flow: {type(node).__name__}")
                self._emit("exec", node)
                calls = [n.func for n in ast.walk(node) if isinstance(n, ast.Call)]
                if any((isinstance(f, ast.Name) and f.id in {"block", "downsample_block", "upsample_block"}) or
                       (isinstance(f, ast.Attribute) and f.attr == "mid_block") for f in calls):
                    self._emit("boundary", ast.unparse(node))

    def start(self, kwargs):
        bound = inspect.signature(self.function).bind(self.module, **kwargs)
        bound.apply_defaults()
        return {"source_sha256": self.source_sha256, "pc": 0, "locals": self.freeze(dict(bound.arguments)),
                "loops": {}, "blocks": 0, "done": False, "output": None}

    def freeze(self, value):
        if value is self.module: return {"$module": ""}
        if isinstance(value, torch.nn.Module):
            if id(value) not in self.module_names: raise BenchError("forward retained an unknown module")
            return {"$module": self.module_names[id(value)]}
        if isinstance(value, torch.dtype): return {"$dtype": str(value).removeprefix("torch.")}
        if isinstance(value, torch.device): return {"$device": str(value)}
        if isinstance(value, dict): return {k: self.freeze(v) for k, v in value.items()}
        if isinstance(value, tuple): return tuple(self.freeze(v) for v in value)
        if isinstance(value, list): return [self.freeze(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool, torch.Tensor)): return value
        raise BenchError(f"native frame retained unsupported closure value {type(value).__name__}")

    def thaw(self, value):
        if isinstance(value, dict):
            if "$module" in value: return self.modules[value["$module"]]
            if "$dtype" in value: return getattr(torch, value["$dtype"])
            if "$device" in value: return torch.device(value["$device"])
            return {k: self.thaw(v) for k, v in value.items()}
        if isinstance(value, tuple): return tuple(self.thaw(v) for v in value)
        if isinstance(value, list): return [self.thaw(v) for v in value]
        return value

    def _eval(self, expression, environment):
        return eval(compile(ast.fix_missing_locations(ast.Expression(expression)), "<native-bench>", "eval"), {**self.globals, **environment}, environment)

    @torch.inference_mode()
    def advance(self, frame, *, finish=False):
        if frame["source_sha256"] != self.source_sha256: raise BenchError("native forward source changed")
        if frame["done"]: raise BenchError("native forward already complete")
        environment = self.thaw(frame["locals"])
        loops = self.thaw(frame["loops"])
        while frame["pc"] < len(self.code):
            op, arg = self.code[frame["pc"]]; frame["pc"] += 1
            if op == "exec":
                exec(compile(ast.fix_missing_locations(ast.Module(body=[arg], type_ignores=[])), "<native-bench>", "exec"), {**self.globals, **environment}, environment)
            elif op == "if":
                if not self._eval(arg[0], environment): frame["pc"] = arg[1]
            elif op == "jump": frame["pc"] = arg
            elif op == "for_init": loops[arg[0]] = {"values": list(self._eval(arg[1], environment)), "index": 0}
            elif op == "for_next":
                key, target, end = arg; loop = loops[key]
                if loop["index"] >= len(loop["values"]):
                    del loops[key]; frame["pc"] = end
                else:
                    environment["__bench_loop_value"] = loop["values"][loop["index"]]
                    assign = ast.Assign(targets=[target], value=ast.Name(id="__bench_loop_value", ctx=ast.Load()))
                    exec(compile(ast.fix_missing_locations(ast.Module(body=[assign], type_ignores=[])), "<native-bench>", "exec"), self.globals, environment)
                    environment.pop("__bench_loop_value"); loop["index"] += 1
            elif op == "boundary":
                frame["blocks"] += 1
                if not finish: break
            elif op == "return":
                frame["output"] = self.freeze(self._eval(arg, environment)); frame["done"] = True; break
        frame["locals"], frame["loops"] = self.freeze(environment), self.freeze(loops)
        return frame
