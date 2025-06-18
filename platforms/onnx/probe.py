"""Probe the ONNX model."""

import argparse
from collections import Counter

import onnx
import onnx.helper


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Probe ONNX model to show comprehensive model information.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('onnx_model_path', type=str,
                        help='Path to the ONNX model file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed information including node details')
    return parser


def get_tensor_info(tensor):
    """Get tensor shape and type information."""
    shape = []
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.HasField('dim_value'):
            shape.append(dim.dim_value)
        elif dim.HasField('dim_param'):
            shape.append(dim.dim_param)
        else:
            shape.append('?')

    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(
        tensor.type.tensor_type.elem_type, 'unknown')

    return shape, dtype.__name__ if hasattr(dtype, '__name__') else str(dtype)


def probe_model(onnx_model_path, verbose=False):
    """Probe ONNX model and return comprehensive information."""
    model = onnx.load(onnx_model_path)

    # Basic model info
    info = {
        'ir_version': model.ir_version,
        'producer_name': model.producer_name,
        'producer_version': model.producer_version,
        'domain': model.domain,
        'model_version': model.model_version,
        'doc_string': model.doc_string
    }

    # Opset versions
    opset_imports = []
    for opset in model.opset_import:
        opset_imports.append(f"{opset.domain or 'ai.onnx'}: {opset.version}")

    # Graph info
    graph = model.graph

    # Input/Output info
    inputs = []
    for inp in graph.input:
        shape, dtype = get_tensor_info(inp)
        inputs.append({
            'name': inp.name,
            'shape': shape,
            'dtype': dtype
        })

    outputs = []
    for out in graph.output:
        shape, dtype = get_tensor_info(out)
        outputs.append({
            'name': out.name,
            'shape': shape,
            'dtype': dtype
        })

    # Operator statistics
    op_counter = Counter()
    for node in graph.node:
        op_counter[node.op_type] += 1

    # Initializer info
    initializers = []
    total_params = 0
    for init in graph.initializer:
        size = 1
        shape = list(init.dims)
        for dim in shape:
            size *= dim
        total_params += size
        initializers.append({
            'name': init.name,
            'shape': shape,
            'size': size,
            'dtype': onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(init.data_type, 'unknown')
        })

    # Node details for verbose mode
    nodes = []
    if verbose:
        for i, node in enumerate(graph.node):
            nodes.append({
                'index': i,
                'op_type': node.op_type,
                'name': node.name or f"node_{i}",
                'inputs': list(node.input),
                'outputs': list(node.output),
                'attributes': {attr.name: onnx.helper.get_attribute_value(attr)
                               for attr in node.attribute}
            })

    return {
        'basic_info': info,
        'opset_imports': opset_imports,
        'inputs': inputs,
        'outputs': outputs,
        'operators': op_counter,
        'initializers': initializers,
        'total_parameters': total_params,
        'total_nodes': len(graph.node),
        'nodes': nodes if verbose else []
    }


def print_model_info(model_info: dict, verbose=False):
    """Print formatted model information."""

    print("=" * 60)
    print("ONNX MODEL PROBE REPORT")
    print("=" * 60)

    # Basic Information
    print("\nBASIC INFORMATION:")
    basic = model_info['basic_info']
    print(f"  IR Version: {basic['ir_version']}")
    print(f"  Producer: {basic['producer_name']} v{basic['producer_version']}")
    print(f"  Domain: {basic['domain'] or 'N/A'}")
    print(f"  Model Version: {basic['model_version']}")
    if basic['doc_string']:
        print(f"  Description: {basic['doc_string']}")

    # Opset Information
    print(f"\nOPSET VERSIONS:")
    for opset in model_info['opset_imports']:
        print(f"  {opset}")

    # Model Structure
    print(f"\nMODEL STRUCTURE:")
    print(f"  Total Nodes: {model_info['total_nodes']}")
    print(f"  Total Parameters: {model_info['total_parameters']:,}")
    print(f"  Initializers: {len(model_info['initializers'])}")

    # Inputs
    print(f"\nINPUTS ({len(model_info['inputs'])}):")
    for inp in model_info['inputs']:
        shape_str = 'x'.join(map(str, inp['shape']))
        print(f"  {inp['name']}: {inp['dtype']}[{shape_str}]")

    # Outputs
    print(f"\nOUTPUTS ({len(model_info['outputs'])}):")
    for out in model_info['outputs']:
        shape_str = 'x'.join(map(str, out['shape']))
        print(f"  {out['name']}: {out['dtype']}[{shape_str}]")

    # Operator Statistics
    print(f"\nOPERATOR STATISTICS:")
    for op, count in sorted(model_info['operators'].items()):
        print(f"  {op}: {count}")

    # Initializers (parameters)
    if model_info['initializers']:
        print(f"\nLARGEST PARAMETERS:")
        sorted_inits = sorted(model_info['initializers'],
                              key=lambda x: x['size'], reverse=True)[:10]
        for init in sorted_inits:
            shape_str = 'x'.join(map(str, init['shape']))
            size_mb = init['size'] * 4 / (1024 * 1024)  # Assuming float32
            print(
                f"  {init['name']}: {shape_str} ({init['size']:,} params, {size_mb:.2f}MB)")

    # Detailed node information
    if verbose and model_info['nodes']:
        print(f"\nDETAILED NODE INFORMATION:")
        for node in model_info['nodes'][:20]:  # Show first 20 nodes
            print(f"  [{node['index']}] {node['op_type']} ({node['name']})")
            print(f"    Inputs: {node['inputs']}")
            print(f"    Outputs: {node['outputs']}")
            if node['attributes']:
                print(f"    Attributes: {list(node['attributes'].keys())}")
            print()

        if len(model_info['nodes']) > 20:
            print(f"  ... and {len(model_info['nodes']) - 20} more nodes")


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    model_info = probe_model(args.onnx_model_path, verbose=args.verbose)
    print_model_info(model_info, verbose=args.verbose)
