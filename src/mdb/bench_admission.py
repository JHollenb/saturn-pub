"""Model OS admission of a payload reference at one bench residual boundary.

The metadata backend retains typed payload references. Its native commit hook is
the only writer of the backend-owned residual. Token publication still belongs
to the VM's token kernel. A receipt never asserts semantic equivalence.
"""
from dataclasses import replace

from mdb.model_os_contracts import (ConsumerContract, ModelPackage, ModelProgram,
    ProgramInstruction, StateSchema, StateSlot, StateSurface)
from mdb.model_os_kernel import InMemoryModelBackend, ModelOSKernel, ModelOSExecutionError

ADDRESS = "bench.residual"


class ResidualBackend(InMemoryModelBackend):
    prepared = None

    def preview(self, parent, plan, updates):
        if self.prepared is not None:
            if set(updates) != {ADDRESS} or updates[ADDRESS] != self.reference:
                raise ModelOSExecutionError("residual payload changed after admission")
            return self.prepared, {"accepted": True, "updated_addresses": [ADDRESS]}
        return super().preview(parent, plan, updates)

    def continue_execution(self, candidate, *, consumer, horizon):
        if horizon != 0 or self.owner.execution is not self.execution:
            raise ModelOSExecutionError("residual boundary moved during preview")
        return {"accepted": True, "consumer_id": consumer.consumer_id,
                "execution_mode": "native-residual-install", "horizon": 0,
                "boundary": self.boundary, "payload": self.reference}

    def commit(self, candidate):
        receipt = super().commit(candidate)
        self.execution.hidden = self.tensor
        return {**receipt, "execution_mode": "native-residual-install"}


def residual_transaction(bench, parent, tensor):
    schema = StateSchema(name="saturn-bench-residual-v1", version=1, slots=(StateSlot(
        address=ADDRESS, value_type="tensor-payload-reference", owner="bench-qwen",
        dtype=str(tensor.dtype), shape=tuple(tensor.shape), read_scopes=("model-os",),
        write_scopes=("model-os",), lifetime="token", branchable=True, persistable=True,
        freshness="candidate", lineage=parent),))
    abi = replace(bench.vm.package.model_abi, state_schema=schema.fingerprint)
    consumer = ConsumerContract(consumer_id="bench.qwen.residual-install", reads=(ADDRESS,),
        output_kind="residual-install-receipt", continuation_tokens=0, native=True)
    program = ModelProgram(program_id="bench.qwen.residual-install.v1", version=1,
        instructions=(ProgramInstruction(instruction_id="install", op="Write",
            reads=(ADDRESS,), writes=(ADDRESS,), outputs=(ADDRESS,), authority="model-os"),),
        portable=False, terminal_claim=False)
    package = ModelPackage(package_id="bench.qwen.residual-install", version=1,
        model_abi=abi, state_schema=schema, state_surface=StateSurface.from_schema(schema,
            surface_id="bench-residual", owner="bench-qwen"), program=program,
        consumers=(consumer,), permissions={"subjects": {"model-os": {
            "read": [ADDRESS], "write": [ADDRESS]}}, "authorities": ["model-os"]})
    backend = ResidualBackend(model_abi=abi, state_schema=schema)
    backend.owner = bench._backend()
    backend.execution = backend.owner.execution
    backend.boundary = {"session_id": bench.session_id, "branch": bench.active,
                        "parent_cut": parent, **bench.cursor()}
    backend.tensor, backend.reference = tensor, bench._tensor_ref(tensor)
    return ModelOSKernel.install(package, backend), {ADDRESS: backend.reference}
