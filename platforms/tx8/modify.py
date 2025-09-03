"""Replace MatMul (matrix by vector, vector by vector) with Unsqueeze + MatMul + Squeeze in ONNX model."""

import argparse
import os

import onnx
import numpy as np
from onnx import helper, numpy_helper


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Replace MatMul (matrix by vector, vector by vector) with Unsqueeze + MatMul + Squeeze in ONNX model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("onnx_model_path", type=str, help="Path to the ONNX model file")
    parser.add_argument(
        "--output_path",
        type=str,
        default="modified_model.onnx",
        help="Path to save the modified ONNX model",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be modified without actually changing the model",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information about modifications",
    )
    return parser


def get_tensor_shape(tensor_name, graph):
    """Get the shape of a tensor from the graph."""
    # Check in value_info
    for value_info in graph.value_info:
        if value_info.name == tensor_name:
            shape = []
            for dim in value_info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                else:
                    shape.append(-1)
            return shape

    # Check in inputs
    for input_info in graph.input:
        if input_info.name == tensor_name:
            shape = []
            for dim in input_info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    shape.append(dim.dim_param)
                else:
                    shape.append(-1)
            return shape

    # Check in initializers
    for init in graph.initializer:
        if init.name == tensor_name:
            return list(init.dims)

    return None


def is_matrix_vector_matmul(node, graph):
    """Check if a MatMul node is matrix by vector multiplication or vector by vector multiplication."""
    if node.op_type != "MatMul":
        return False

    if len(node.input) != 2:
        return False

    input1_shape = get_tensor_shape(node.input[0], graph)
    input2_shape = get_tensor_shape(node.input[1], graph)

    if input1_shape is None or input2_shape is None:
        return False

    # Remove batch dimensions and unknown dimensions for analysis
    def get_effective_shape(shape):
        return [dim for dim in shape if isinstance(dim, int) and dim > 0]

    eff_shape1 = get_effective_shape(input1_shape)
    eff_shape2 = get_effective_shape(input2_shape)

    # Matrix by vector: [M, N] x [N] or [batch, M, N] x [N]
    if len(eff_shape1) >= 2 and len(eff_shape2) == 1:
        if eff_shape1[-1] == eff_shape2[0]:
            return True, "matrix_vector"

    # Vector by matrix: [N] x [N, M]
    if len(eff_shape1) == 1 and len(eff_shape2) >= 2:
        if eff_shape1[0] == eff_shape2[-2]:
            return True, "vector_matrix"

    # Vector by vector (dot product): [N] x [N] -> scalar
    if len(eff_shape1) == 1 and len(eff_shape2) == 1:
        if eff_shape1[0] == eff_shape2[0]:
            return True, "vector_vector"

    return False, None


def create_axes_initializer(axes, name):
    """Create an initializer for axes tensor."""
    axes_array = np.array(axes, dtype=np.int64)
    axes_tensor = numpy_helper.from_array(axes_array, name)
    return axes_tensor


def create_replacement_nodes(matmul_node, graph, node_counter):
    """Create replacement nodes for MatMul operation."""
    input1_name = matmul_node.input[0]
    input2_name = matmul_node.input[1]
    output_name = matmul_node.output[0]

    is_replacement, mult_type = is_matrix_vector_matmul(matmul_node, graph)
    if not is_replacement:
        return None, [], node_counter

    new_nodes = []
    new_initializers = []
    base_name = matmul_node.name or f"matmul_{node_counter}"

    if mult_type == "matrix_vector":
        # Matrix [M, N] x Vector [N] -> [M]
        # Transform to: Matrix [M, N] x Vector [N, 1] -> [M, 1] -> [M]

        # Create axes initializers
        unsqueeze_axes_name = f"{base_name}_unsqueeze_axes"
        squeeze_axes_name = f"{base_name}_squeeze_axes"

        unsqueeze_axes_tensor = create_axes_initializer([1], unsqueeze_axes_name)
        squeeze_axes_tensor = create_axes_initializer([1], squeeze_axes_name)

        new_initializers.extend([unsqueeze_axes_tensor, squeeze_axes_tensor])

        # Step 1: Unsqueeze vector to [N, 1]
        unsqueeze_output = f"{base_name}_unsqueezed_vector"
        unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs=[input2_name, unsqueeze_axes_name],
            outputs=[unsqueeze_output],
            name=f"{base_name}_unsqueeze",
        )
        new_nodes.append(unsqueeze_node)

        # Step 2: MatMul with unsqueezed vector
        matmul_output = f"{base_name}_matmul_result"
        matmul_node_new = helper.make_node(
            "MatMul",
            inputs=[input1_name, unsqueeze_output],
            outputs=[matmul_output],
            name=f"{base_name}_matmul",
        )
        new_nodes.append(matmul_node_new)

        # Step 3: Squeeze result to remove added dimension
        squeeze_node = helper.make_node(
            "Squeeze",
            inputs=[matmul_output, squeeze_axes_name],
            outputs=[output_name],
            name=f"{base_name}_squeeze",
        )
        new_nodes.append(squeeze_node)

    elif mult_type == "vector_matrix":
        # Vector [N] x Matrix [N, M] -> [M]
        # Transform to: Vector [1, N] x Matrix [N, M] -> [1, M] -> [M]

        # Create axes initializers
        unsqueeze_axes_name = f"{base_name}_unsqueeze_axes"
        squeeze_axes_name = f"{base_name}_squeeze_axes"

        unsqueeze_axes_tensor = create_axes_initializer([0], unsqueeze_axes_name)
        squeeze_axes_tensor = create_axes_initializer([0], squeeze_axes_name)

        new_initializers.extend([unsqueeze_axes_tensor, squeeze_axes_tensor])

        # Step 1: Unsqueeze vector to [1, N]
        unsqueeze_output = f"{base_name}_unsqueezed_vector"
        unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs=[input1_name, unsqueeze_axes_name],
            outputs=[unsqueeze_output],
            name=f"{base_name}_unsqueeze",
        )
        new_nodes.append(unsqueeze_node)

        # Step 2: MatMul with unsqueezed vector
        matmul_output = f"{base_name}_matmul_result"
        matmul_node_new = helper.make_node(
            "MatMul",
            inputs=[unsqueeze_output, input2_name],
            outputs=[matmul_output],
            name=f"{base_name}_matmul",
        )
        new_nodes.append(matmul_node_new)

        # Step 3: Squeeze result to remove added dimension
        squeeze_node = helper.make_node(
            "Squeeze",
            inputs=[matmul_output, squeeze_axes_name],
            outputs=[output_name],
            name=f"{base_name}_squeeze",
        )
        new_nodes.append(squeeze_node)

    elif mult_type == "vector_vector":
        # Vector [N] x Vector [N] -> scalar (dot product)
        # Transform to: Vector [1, N] x Vector [N, 1] -> [1, 1] -> scalar

        # Create axes initializers
        unsqueeze1_axes_name = f"{base_name}_unsqueeze1_axes"
        unsqueeze2_axes_name = f"{base_name}_unsqueeze2_axes"
        squeeze_axes_name = f"{base_name}_squeeze_axes"

        unsqueeze1_axes_tensor = create_axes_initializer(
            [0], unsqueeze1_axes_name
        )  # [N] -> [1, N]
        unsqueeze2_axes_tensor = create_axes_initializer(
            [1], unsqueeze2_axes_name
        )  # [N] -> [N, 1]
        squeeze_axes_tensor = create_axes_initializer(
            [0, 1], squeeze_axes_name
        )  # [1, 1] -> scalar

        new_initializers.extend(
            [unsqueeze1_axes_tensor, unsqueeze2_axes_tensor, squeeze_axes_tensor]
        )

        # Step 1: Unsqueeze first vector to [1, N]
        unsqueeze1_output = f"{base_name}_unsqueezed_vector1"
        unsqueeze1_node = helper.make_node(
            "Unsqueeze",
            inputs=[input1_name, unsqueeze1_axes_name],
            outputs=[unsqueeze1_output],
            name=f"{base_name}_unsqueeze1",
        )
        new_nodes.append(unsqueeze1_node)

        # Step 2: Unsqueeze second vector to [N, 1]
        unsqueeze2_output = f"{base_name}_unsqueezed_vector2"
        unsqueeze2_node = helper.make_node(
            "Unsqueeze",
            inputs=[input2_name, unsqueeze2_axes_name],
            outputs=[unsqueeze2_output],
            name=f"{base_name}_unsqueeze2",
        )
        new_nodes.append(unsqueeze2_node)

        # Step 3: MatMul [1, N] x [N, 1] -> [1, 1]
        matmul_output = f"{base_name}_matmul_result"
        matmul_node_new = helper.make_node(
            "MatMul",
            inputs=[unsqueeze1_output, unsqueeze2_output],
            outputs=[matmul_output],
            name=f"{base_name}_matmul",
        )
        new_nodes.append(matmul_node_new)

        # Step 4: Squeeze result to scalar
        squeeze_node = helper.make_node(
            "Squeeze",
            inputs=[matmul_output, squeeze_axes_name],
            outputs=[output_name],
            name=f"{base_name}_squeeze",
        )
        new_nodes.append(squeeze_node)

    return new_nodes, new_initializers, node_counter + 1


def modify_model(model_path, output_path, dry_run=False, verbose=False):
    """Modify the ONNX model by replacing MatMul operations."""
    # Load the model
    model = onnx.load(model_path)
    graph = model.graph

    # Find MatMul nodes that need replacement
    nodes_to_replace = []
    replacement_info = []

    for i, node in enumerate(graph.node):
        if node.op_type == "MatMul":
            is_replacement, mult_type = is_matrix_vector_matmul(node, graph)
            if is_replacement:
                nodes_to_replace.append((i, node))
                input1_shape = get_tensor_shape(node.input[0], graph)
                input2_shape = get_tensor_shape(node.input[1], graph)
                replacement_info.append(
                    {
                        "node_name": node.name or f"node_{i}",
                        "type": mult_type,
                        "input1_shape": input1_shape,
                        "input2_shape": input2_shape,
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                    }
                )

    if verbose or dry_run:
        print(f"Found {len(nodes_to_replace)} MatMul nodes to replace:")
        for info in replacement_info:
            print(f"  - {info['node_name']}: {info['type']}")
            print(f"    Input shapes: {info['input1_shape']} x {info['input2_shape']}")
            print(f"    Inputs: {info['inputs']}")
            print(f"    Outputs: {info['outputs']}")
            print()

    if dry_run:
        print("Dry run completed. No changes made to the model.")
        return

    if not nodes_to_replace:
        print("No MatMul (matrix by vector) operations found to replace.")
        if output_path != model_path:
            # Still save a copy if output path is different
            onnx.save(model, output_path)
            print(f"Original model saved to: {output_path}")
        return

    # Prepare replacement operations
    # List of (original_index, replacement_nodes, replacement_initializers)
    replacements = []
    node_counter = 0

    for original_index, original_node in nodes_to_replace:
        replacement_nodes, replacement_initializers, node_counter = (
            create_replacement_nodes(original_node, graph, node_counter)
        )

        if replacement_nodes:
            replacements.append(
                (original_index, replacement_nodes, replacement_initializers)
            )
            if verbose:
                print(
                    f"Prepared replacement for node {original_node.name or f'node_{original_index}'} "
                    f"with {len(replacement_nodes)} nodes"
                )

    # Apply replacements by modifying the graph in-place
    # Sort replacements by index in reverse order to avoid index shifting issues
    replacements.sort(key=lambda x: x[0], reverse=True)

    nodes_replaced = 0
    total_initializers_added = 0

    for original_index, replacement_nodes, replacement_initializers in replacements:
        # Remove the original node
        del graph.node[original_index]

        # Insert replacement nodes at the same position
        for i, new_node in enumerate(replacement_nodes):
            graph.node.insert(original_index + i, new_node)

        # Add new initializers
        graph.initializer.extend(replacement_initializers)
        total_initializers_added += len(replacement_initializers)

        nodes_replaced += 1

        if verbose:
            print(
                f"Replaced node at index {original_index} with {len(replacement_nodes)} nodes"
            )

    # Set target IR version for compatibility
    target_ir_version = 10
    model.ir_version = target_ir_version

    # Perform shape inference on the modified model
    if verbose:
        print("🔍 Performing shape inference...")

    try:
        # Shape inference to populate value_info with intermediate tensor shapes
        inferred_model = onnx.shape_inference.infer_shapes(model)

        # Replace the original model with the inferred one
        model = inferred_model

        if verbose:
            print("✅ Shape inference completed successfully")
            print(f"   Value info entries: {len(model.graph.value_info)}")

            # Show some shape inference results
            if len(model.graph.value_info) > 0:
                print("   Sample inferred shapes:")
                # Show first 5
                for i, value_info in enumerate(model.graph.value_info[:5]):
                    shape = []
                    for dim in value_info.type.tensor_type.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        elif dim.HasField("dim_param"):
                            shape.append(dim.dim_param)
                        else:
                            shape.append("?")
                    shape_str = "x".join(map(str, shape))
                    dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE.get(
                        value_info.type.tensor_type.elem_type, "unknown"
                    )
                    dtype_name = (
                        dtype.__name__ if hasattr(dtype, "__name__") else str(dtype)
                    )
                    print(f"     {value_info.name}: {dtype_name}[{shape_str}]")

                if len(model.graph.value_info) > 5:
                    print(f"     ... and {len(model.graph.value_info) - 5} more")

    except Exception as e:
        print(f"⚠️  Shape inference warning: {e}")
        print("   Continuing without shape inference...")

    # Validate the modified model
    try:
        onnx.checker.check_model(model)
        print("✅ Model validation passed")
    except Exception as e:
        print(f"⚠️  Model validation warning: {e}")

    # Save the modified model
    onnx.save(model, output_path)

    print("\n🎉 Model modification completed:")
    print(f"  - Replaced {nodes_replaced} MatMul operations")
    print(f"  - Added {total_initializers_added} new initializers")
    print(f"  - Total nodes: {len(model.graph.node)}")
    print(f"  - Total initializers: {len(model.graph.initializer)}")
    print(f"  - Value info entries: {len(model.graph.value_info)}")
    print(f"  - IR version: {model.ir_version}")
    print(f"  - Saved to: {output_path}")


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if not os.path.exists(args.onnx_model_path):
        print(f"❌ Error: Model file not found: {args.onnx_model_path}")
        exit(1)

    try:
        modify_model(
            args.onnx_model_path,
            args.output_path,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"❌ Error modifying model: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
