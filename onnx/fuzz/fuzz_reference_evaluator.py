# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Atheris fuzz harness for onnx.reference.ReferenceEvaluator.

The reference implementation is pure Python and covers every operator, so it is
not exercised by the harnesses that target the C++ checker, parser and shape
inference. A fuzzer-controlled toggle byte at the tail selects the input path:

* Raw bytes  -> onnx.load_model_from_string -> ReferenceEvaluator(...)
  Reaches operator resolution, subgraph handling and the shape annotation
  checks on crafted serialized models, without running any kernel.

* Structured -> single-node model built from FuzzedDataProvider -> run()
  Picks a registered operator schema (default and ai.onnx.ml domains), builds
  small tensors of fuzzer-chosen dtype and shape plus generated attributes,
  and executes the node, so the per-operator kernels in onnx/reference/ops are
  reached with well-formed models.

The reference evaluator is not hardened against invalid operands: it raises
ordinary Python exceptions such as ValueError, TypeError or IndexError for
inputs an operator does not accept. Those are expected and are ignored, so a
report means an exception escaped a layer that should have converted it, or a
hang, out-of-memory condition or interpreter-level fault. Tensor sizes are kept
small to bound the cost of each run.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import atheris

# numpy is imported outside instrument_imports(): it refuses to be loaded
# twice, which instrumentation can trigger for C-extension packages.
import numpy as np

if TYPE_CHECKING:
    from typing import Any

with atheris.instrument_imports():
    import onnx
    from onnx import TensorProto, defs, helper
    from onnx.reference import ReferenceEvaluator

_DOMAINS = ("", "ai.onnx.ml")

# Latest version of every operator in the default and ML domains.
_SCHEMAS = tuple(
    schema
    for schema in defs.get_all_schemas()
    if schema.domain in _DOMAINS and schema.name
)

_DTYPES = (
    np.float32,
    np.float64,
    np.float16,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.bool_,
    np.str_,
)

_MAX_RANK = 4
_MAX_DIM = 4
_MAX_INPUTS = 6


def _make_array(fdp, like=None):
    # Most operators need operands whose dtype or shape agree, which independent
    # random choices rarely produce, so an input can copy an earlier one.
    if like is not None and fdp.ConsumeBool():
        dtype = like.dtype.type
        shape = like.shape
    else:
        # Floating-point tensors drive most kernels, so favour float32.
        if fdp.ConsumeIntInRange(0, 2) == 0:
            dtype = np.float32
        else:
            dtype = _DTYPES[fdp.ConsumeIntInRange(0, len(_DTYPES) - 1)]
        rank = fdp.ConsumeIntInRange(0, _MAX_RANK)
        # Empty dimensions are valid but rarely interesting, so keep them rare.
        shape = tuple(
            0
            if fdp.ConsumeIntInRange(0, 9) == 0
            else fdp.ConsumeIntInRange(1, _MAX_DIM)
            for _ in range(rank)
        )
    size = int(np.prod(shape, dtype=np.int64))
    if dtype is np.str_:
        values = [
            ("a", "bc", "Hello World", "")[fdp.ConsumeIntInRange(0, 3)]
            for _ in range(size)
        ]
    elif dtype is np.bool_:
        values = [fdp.ConsumeBool() for _ in range(size)]
    elif np.issubdtype(dtype, np.floating):
        values = [fdp.ConsumeFloatInRange(-8.0, 8.0) for _ in range(size)]
    else:
        info = np.iinfo(dtype)
        low = max(int(info.min), -8)
        high = min(int(info.max), 8)
        values = [fdp.ConsumeIntInRange(low, high) for _ in range(size)]
    return np.array(values, dtype=dtype).reshape(shape)


def _small_ints(fdp):
    return [fdp.ConsumeIntInRange(-2, 4) for _ in range(fdp.ConsumeIntInRange(0, 3))]


# Attributes that hold one entry per spatial axis (or two per axis for pads).
_SPATIAL_ATTRS = frozenset(
    {"kernel_shape", "strides", "dilations", "pads", "output_padding", "output_shape"}
)


def _make_attribute(fdp, name, attr_type, first=None) -> Any:
    t = defs.OpSchema.AttrType
    if attr_type == t.INT:
        value: Any = fdp.ConsumeIntInRange(-2, 4)
    elif attr_type == t.INTS:
        if name in _SPATIAL_ATTRS and first is not None and first.ndim >= 3:
            # Well-formed convolution and pooling attributes, which random
            # lengths and signs almost never produce.
            axes = first.ndim - 2
            count = 2 * axes if name == "pads" else axes
            low = 0 if name in ("pads", "output_padding") else 1
            value = [fdp.ConsumeIntInRange(low, 3) for _ in range(count)]
        else:
            value = _small_ints(fdp)
    elif attr_type == t.FLOAT:
        value = float(fdp.ConsumeIntInRange(-2, 4))
    elif attr_type == t.FLOATS:
        value = [float(v) for v in _small_ints(fdp)]
    elif attr_type == t.STRING:
        value = ["a", "b", "NOTSET", "constant", "nearest", "SUM"][
            fdp.ConsumeIntInRange(0, 5)
        ]
    elif attr_type == t.STRINGS:
        value = ["a", "b"][: fdp.ConsumeIntInRange(0, 2)]
    elif attr_type == t.TENSOR:
        value = helper.make_tensor(name, TensorProto.FLOAT, [1], [1.0])
    elif attr_type == t.GRAPH:
        value = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            name,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, None)],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
    else:
        return None
    return helper.make_attribute(name, value)


def _build_node_model(fdp):
    schema = _SCHEMAS[fdp.ConsumeIntInRange(0, len(_SCHEMAS) - 1)]
    max_inputs = min(schema.max_input, _MAX_INPUTS)
    count = fdp.ConsumeIntInRange(schema.min_input, max(schema.min_input, max_inputs))
    feeds: dict[str, np.ndarray] = {}
    first = None
    input_names: list[str] = []
    graph_inputs = []
    for i in range(count):
        # Optional inputs may be omitted by naming them with an empty string.
        if i >= schema.min_input and fdp.ConsumeBool():
            input_names.append("")
            continue
        name = f"x{i}"
        array = _make_array(fdp, first)
        first = first if first is not None else array
        feeds[name] = array
        input_names.append(name)
        graph_inputs.append(
            helper.make_tensor_value_info(
                name,
                helper.np_dtype_to_tensor_dtype(array.dtype),
                list(array.shape),
            )
        )
    n_outputs = max(1, min(schema.max_output, 3))
    output_names = [f"y{i}" for i in range(n_outputs)]
    attributes = []
    for attr_name, attr in schema.attributes.items():
        if attr.required or fdp.ConsumeBool():
            made = _make_attribute(fdp, attr_name, attr.type, first)
            if made is not None:
                attributes.append(made)
    node = helper.make_node(
        schema.name, input_names, output_names, domain=schema.domain
    )
    node.attribute.extend(attributes)
    graph = helper.make_graph(
        [node],
        "g",
        graph_inputs,
        [helper.make_empty_tensor_value_info(n) for n in output_names],
    )
    opsets = [helper.make_opsetid("", defs.onnx_opset_version())]
    if schema.domain:
        opsets.append(helper.make_opsetid(schema.domain, schema.since_version))
    model = helper.make_model(graph, opset_imports=opsets)
    return model, feeds


def _run_structured(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    try:
        model, feeds = _build_node_model(fdp)
    except Exception:
        return
    try:
        ReferenceEvaluator(model).run(None, feeds)
    except Exception:
        return


def _run_raw(data: bytes) -> None:
    try:
        model = onnx.load_model_from_string(data)
    except Exception:
        return
    try:
        ReferenceEvaluator(model)
    except Exception:
        return


def TestOneInput(data: bytes) -> None:
    if not data:
        return
    body, toggle = data[:-1], data[-1]
    if toggle & 1:
        _run_raw(body)
    else:
        _run_structured(body)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
