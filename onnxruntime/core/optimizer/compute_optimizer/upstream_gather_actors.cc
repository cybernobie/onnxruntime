// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE
#include <onnx/defs/attr_proto_util.h>

#include "core/common/safeint.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/upstream_gather_actors.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime::optimizer::compute_optimizer {

enum class ShapeCompareRet {
  ExactEqual = 0,
  BroadcastableEqual = 1,
  RankTooLow = 2,
  NotEqual = 3,
  ShapeCompareRetMax = 4,
};

/**
 * @brief Check dimensions are equal or broadcastable before axis.
 *
 * @param full_broadcasted_shape Full broadcasted shape as a baseline to compare.
 * @param axis The axis (inclusive, of full_broadcasted_shape) where we end the comparison.
 * @param target_shape Shape to compare, can have dim value be 1 for broadcastable dimension.
 * @return A pair of bool, bool. The first bool is true if the dimensions are exactly same before and include axis.
 * The second bool is true if the dimension of target_shape has dim value be 1 on axis.
 */
std::pair<ShapeCompareRet, bool> AreDimsCompatibleBeforeAxisInternal(
    const TensorShapeProto* full_broadcasted_shape, const int axis,
    const TensorShapeProto* target_shape) {
  int full_rank = full_broadcasted_shape->dim_size();
  int target_rank = target_shape->dim_size();

  ORT_ENFORCE(full_rank >= axis && target_rank <= full_rank, "full_rank should bigger than axis and target_rank ",
              axis, " full_rank: ", full_rank, " target_rank: ", target_rank);

  int minimum_rank_to_handle = full_rank - axis;
  if (target_rank < minimum_rank_to_handle) {
    // Skip if target node's input rank is less than minimum rank to handle.
    // Essentially this means the input did not affect the Gather axis.
    return std::make_pair(ShapeCompareRet::RankTooLow, false);
  }

  bool exact_equal = true;
  bool broadcastable_equal = true;
  bool dim_be_1_on_axis = false;

  int axis_iter = axis;
  int negative_axis = axis < 0 ? axis : axis - full_rank;
  int target_axis_iter = target_rank + negative_axis;

  for (; axis_iter >= 0 && target_axis_iter >= 0; --axis_iter, --target_axis_iter) {
    auto& dim = full_broadcasted_shape->dim(axis_iter);
    auto& target_dim = target_shape->dim(target_axis_iter);
    if (dim.has_dim_value() && target_dim.has_dim_value()) {
      if (dim.dim_value() != target_dim.dim_value()) {
        exact_equal = false;
        if (target_dim.dim_value() == 1) {
          if (axis_iter == axis) dim_be_1_on_axis = true;
        } else {
          broadcastable_equal = false;
        }
      }
    } else if (dim.has_dim_param() && target_dim.has_dim_param()) {
      if (dim.dim_param() != target_dim.dim_param()) {
        exact_equal = false;
      }
    } else {
      exact_equal = false;
      if (target_dim.has_dim_value() && target_dim.dim_value() == 1) {
        if (axis_iter == axis) dim_be_1_on_axis = true;
      } else {
        broadcastable_equal = false;
      }
    }
  }

  if (exact_equal) {
    return std::make_pair(ShapeCompareRet::ExactEqual, dim_be_1_on_axis);
  } else if (broadcastable_equal) {
    return std::make_pair(ShapeCompareRet::BroadcastableEqual, dim_be_1_on_axis);
  } else {
    return std::make_pair(ShapeCompareRet::NotEqual, dim_be_1_on_axis);
  }
}

/**
 * @brief Check input meet pass through requirement.
 *
 * @param current_node_output_arg_to_gather The output arg of current node that consumed by slice node.
 * @param arg_to_compare The input/output arg to check.
 * @param info Slice info.
 * @param logger The logger.
 * @param fatal_error_found Used as return value. If fatal error found, set to true. Fatal error means,
 *   we cannot pass through this input arg.
 * @param dim_1_for_axis_found Used as return value. If dim value is 1 for axis, set to true.
 * @return a int represent the new slice axis for the input arg, if pass through needed to be done for
 * this input arg, otherwise, return nullptr.
 *
 * For each input of current_node, using this function to check if the input can be passed through.
 * If the input has dim on negative_axis and
 * 1). either the dimension (if exists) including and before negative_axis is same as target node's output shape.
 * 2). or the dimension (if exists) including and before negative_axis is 1.
 * Otherwise, we will skip the optimization.
 *
 * Example 1: [Can be passed through]
 *    input_0 [M, N, K]    input_1 [K]
 *                \        /
 *                Add [M, N, K] (current_node)
 *                     |
 *            Gather0(axis=1, indices=[1])
 *                     |
 *              output [M, 1, K]
 * In this case, we can propagate Gather to input_0 branch, input_1 is skipped because it did not has dim on
 * slicing axis.
 *
 * Example 2: [Can be passed through]
 *    input_0 [M, N, K]    input_1 [N, K]
 *                \        /
 *                Add [M, N, K] (current_node)
 *                     |
 *            Gather0(axis=1, indices=[1])
 *                     |
 *              output [M, 1, K]
 * In this case, we can propagate Gather to input_0 and input-1 branch, because including and before slicing axis 1,
 * all dims are equal.
 *
 * Example 3: [Can be passed through]
 *    input_0 [M, N, K]    input_1 [1, K]
 *                \        /
 *                Add [M, N, K] (current_node)
 *                     |
 *            Gather0(axis=1, indices=[1])
 *                     |
 *              output [M, 1, K]
 * In this case, we can propagate Gather to input_0 branch, input_1 branch is skipped because it has dim 1 on slicing
 * axis.
 *
 * Example 4: [Can be passed through]
 *    input_0 [M, N, K]    input_1 [1, N, K]
 *                \        /
 *                Add [M, N, K] (current_node)
 *                     |
 *            Gather0(axis=1, indices=[1])
 *                     |
 *              output [M, 1, K]
 * In this case, we can propagate Gather to input_0 and input_1 branch.
 *
 * Example 5: [Can be passed through]
 *    input_0 [M, N, K]    input_1 [M, 1, K]
 *                \        /
 *                Add [M, N, K] (current_node)
 *                     |
 *            Gather0(axis=1, indices=[1])
 *                     |
 *              output [M, 1, K]
 * In this case, we can propagate Gather to input_0 branch, input_1 branch is skipped because it has dim 1 on slicing.
 *
 * Example 6: [CANNOT be passed through]
 *    input_0 [M, N, K]    input_1 [L, N, K]
 *                \        /
 *                Add [M, N, K] (current_node)
 *                     |
 *            Gather0(axis=1, indices=[1])
 *                     |
 *              output [M, 1, K]
 *
 */
std::optional<int> CheckInputForPassThrough(const NodeArg* current_node_output_arg_to_gather,
                                            const NodeArg* arg_to_compare,
                                            const SliceInfo& info,
                                            const logging::Logger& logger,
                                            bool& fatal_error_found,
                                            bool& dim_1_for_axis_found) {
  fatal_error_found = false;
  auto ret_pair = AreDimsCompatibleBeforeAxisInternal(current_node_output_arg_to_gather->Shape(),
                                                      info.non_negative_axis,
                                                      arg_to_compare->Shape());
  if (ret_pair.first == ShapeCompareRet::ExactEqual) {
    return info.non_negative_axis;
  } else if (ret_pair.first == ShapeCompareRet::RankTooLow) {
    LOG_DEBUG_INFO(logger, "Skip " + arg_to_compare->Name() + " because its rank is too low.");
    return std::nullopt;
  } else if (ret_pair.first == ShapeCompareRet::NotEqual) {
    fatal_error_found = true;
    return std::nullopt;
  } else if (ret_pair.first == ShapeCompareRet::BroadcastableEqual) {
    if (ret_pair.second) {
      LOG_DEBUG_INFO(logger, "Skip " + arg_to_compare->Name() +
                                 ", whose dim on axis is 1, no need to Gather from.");
      dim_1_for_axis_found = true;
      return std::nullopt;
    }
    return info.non_negative_axis;
  }

  ORT_THROW("Unexpected return value from CheckInputForPassThrough.");
}

TensorShapeProto CreateTensorShapeInsertDimAtAxis(const TensorShapeProto* src_shape, int axis, int64_t dim_value) {
  ORT_ENFORCE(axis <= src_shape->dim_size(), "axis is out of range.", axis, " vs ", src_shape->dim_size());
  TensorShapeProto updated_shape;
  int j = 0;
  for (j = 0; j < axis; ++j) {
    auto dim = src_shape->dim(j);
    if (dim.has_dim_value()) {
      updated_shape.add_dim()->set_dim_value(dim.dim_value());
    } else if (dim.has_dim_param()) {
      updated_shape.add_dim()->set_dim_param(dim.dim_param());
    } else {
      ORT_THROW("Invalid dim found in CreateTensorShapeInsertDimAtAxis");
    }
  }
  updated_shape.add_dim()->set_dim_value(dim_value);
  for (; j < src_shape->dim_size(); ++j) {
    auto dim = src_shape->dim(j);
    if (dim.has_dim_value()) {
      updated_shape.add_dim()->set_dim_value(dim.dim_value());
    } else if (dim.has_dim_param()) {
      updated_shape.add_dim()->set_dim_param(dim.dim_param());
    } else {
      ORT_THROW("Invalid dim found in CreateTensorShapeInsertDimAtAxis");
    }
  }
  return updated_shape;
}

void AdaptInputAndOutputForScalarSlice(Graph& graph, Node& current_node, int current_node_output_index,
                                       int slice_axis, const std::string& entry_node_name,
                                       const std::unordered_map<int, SliceInfo>& new_gather_infos,
                                       const logging::Logger& logger) {
  LOG_DEBUG_INFO(logger, "AdaptInputAndOutputForScalarSlice for Node " + current_node.Name() + "(" +
                             current_node.OpType() + ")");

  // For each handled inputs, insert Unsqueeze node to get the removed dim back at slice_axis.
  for (auto pair : new_gather_infos) {
    int input_index = pair.first;
    Node* new_node = nullptr;
    // Be noted, the Unsqueeze should happens on the axis of new slice node.
    if (GetONNXOpSetVersion(graph) < 13) {
      onnxruntime::NodeAttributes attributes;
      attributes["axes"] = ONNX_NAMESPACE::MakeAttribute("axes", std::vector<int64_t>{pair.second.non_negative_axis});

      new_node =
          InsertIntermediateNodeOnDestInput(
              graph,
              current_node, input_index,
              0 /* new node input index to connect to current_node's input node*/,
              0 /* new node output index to connect to current_node*/,
              graph.GenerateNodeName(entry_node_name + "_adapt_input"),
              "Unsqueeze",
              "Unsqueeze node",
              {current_node.MutableInputDefs()[input_index]},
              {&graph.GetOrCreateNodeArg(
                  graph.GenerateNodeArgName("unsqueeze_adaptor"),
                  current_node.MutableInputDefs()[input_index]->TypeAsProto())},
              attributes, kOnnxDomain,
              logger);
    } else {
      new_node =
          InsertIntermediateNodeOnDestInput(
              graph,
              current_node, input_index,
              0 /* new node input index to connect to current_node's input node*/,
              0 /* new node output index to connect to current_node*/,
              graph.GenerateNodeName(entry_node_name + "_adapt_input"),
              "Unsqueeze",
              "Unsqueeze node",
              {current_node.MutableInputDefs()[input_index],
               CreateUnsqueezeAxesInitializer(graph, {pair.second.non_negative_axis})},
              {&graph.GetOrCreateNodeArg(
                  graph.GenerateNodeArgName("unsqueeze_adaptor"),
                  current_node.MutableInputDefs()[input_index]->TypeAsProto())},
              {}, kOnnxDomain,
              logger);
    }
    new_node->SetExecutionProviderType(current_node.GetExecutionProviderType());
    // Set correct shape for Unsqueeze node
    const TensorShapeProto* unsqueeze_input_shape = new_node->MutableInputDefs()[0]->Shape();
    new_node->MutableOutputDefs()[0]->SetShape(
        CreateTensorShapeInsertDimAtAxis(unsqueeze_input_shape, pair.second.non_negative_axis, 1));
  }

  // Find the consumer node of MatMul, and the input index of that node connect to MatMul.
  std::vector<const Node*> consumers =
      graph.GetConsumerNodes(current_node.MutableOutputDefs()[current_node_output_index]->Name());
  ORT_ENFORCE(consumers.size() >= 1, "MatMul should have at least one consumer at this point. " +
                                         std::to_string(consumers.size()) + " consumers found.");
  Node& consumer = *graph.GetNode(consumers[0]->Index());
  int index = -1;
  for (size_t i = 0; i < consumer.InputDefs().size(); ++i) {
    auto input_arg = consumer.InputDefs()[i];
    if (input_arg->Name().compare(current_node.MutableOutputDefs()[current_node_output_index]->Name()) == 0) {
      index = static_cast<int>(i);
      break;
    }
  }

  // Create Squeeze node connecting MatMul output to consumer node.
  Node* matmul_out_adaptor_node = nullptr;
  if (GetONNXOpSetVersion(graph) < 13) {
    onnxruntime::NodeAttributes attributes;
    attributes["axes"] = ONNX_NAMESPACE::MakeAttribute("axes", std::vector<int64_t>{slice_axis});
    matmul_out_adaptor_node =
        InsertIntermediateNodeOnDestInput(
            graph, consumer, index,
            0,
            0 /* new node output index*/,
            graph.GenerateNodeName(current_node.OpType() + "_output"),
            "Squeeze",
            "Squeeze node",
            {consumer.MutableInputDefs()[index]},
            {&graph.GetOrCreateNodeArg(
                graph.GenerateNodeArgName("squeeze_adaptor"),
                consumer.MutableInputDefs()[index]->TypeAsProto())},
            attributes, kOnnxDomain, logger);
  } else {
    matmul_out_adaptor_node =
        InsertIntermediateNodeOnDestInput(
            graph, consumer, index,
            0,
            0 /* new node output index*/,
            graph.GenerateNodeName(current_node.OpType() + "_output"),
            "Squeeze",
            "Squeeze node",
            {consumer.MutableInputDefs()[index],
             CreateUnsqueezeAxesInitializer(graph, {slice_axis})},
            {&graph.GetOrCreateNodeArg(
                graph.GenerateNodeArgName("squeeze_adaptor"),
                consumer.MutableInputDefs()[index]->TypeAsProto())},
            {}, kOnnxDomain, logger);
  }

  matmul_out_adaptor_node->SetExecutionProviderType(current_node.GetExecutionProviderType());

  // Don't need set shape for Squeeze because original MatMul output is used as its output type.
  // Set correct shape for MatMul node
  const TensorShapeProto* matmul_out_shape = matmul_out_adaptor_node->MutableOutputDefs()[0]->Shape();
  current_node.MutableOutputDefs()[0]->SetShape(CreateTensorShapeInsertDimAtAxis(matmul_out_shape, slice_axis, 1));
}

bool DefaultUpStreamGatherOperatorActorBase::PostProcess(
    Graph& graph, Node& current_node, int current_node_output_index,
    int slice_axis, bool is_slice_scalar, bool input_has_dim_1_for_axis,
    const ONNX_NAMESPACE::TensorShapeProto_Dimension& /*output_dim_on_axis*/,
    const std::string& entry_node_name,
    const std::unordered_map<int, SliceInfo>& new_gather_infos,
    const logging::Logger& logger) {
  LOG_DEBUG_INFO(logger, "Enter DefaultUpStreamGatherOperatorActorBase::PostProcess for Node " + current_node.Name() +
                             "(" + current_node.OpType() + ")");
  if (is_slice_scalar && input_has_dim_1_for_axis) {
    AdaptInputAndOutputForScalarSlice(graph, current_node, current_node_output_index, slice_axis,
                                      entry_node_name, new_gather_infos, logger);
  }

  return true;
}

bool SimplePassThroughActor::PreCheck(const Graph& /*graph*/, const Node& current_node, const SliceInfo& info,
                                      const std::vector<int>& allowed_input_indices,
                                      const logging::Logger& logger,
                                      std::unordered_map<int, int>& propagate_input_config,
                                      bool& input_has_dim_1_for_axis) {
  LOG_DEBUG_INFO(logger, "Enter SimplePassThroughActor::PreCheck for node " + current_node.Name());

  Node* slice_node = info.node_ptr;
  int current_node_output_index = optimizer_utils::IndexOfNodeOutput(current_node, *slice_node->InputDefs()[0]);
  const NodeArg* gather_data_input_arg = current_node.OutputDefs()[current_node_output_index];

  propagate_input_config.clear();
  input_has_dim_1_for_axis = false;
  for (size_t i = 0; i < current_node.InputDefs().size(); ++i) {
    if (allowed_input_indices.size() > 0 &&
        std::find(allowed_input_indices.begin(), allowed_input_indices.end(), i) == allowed_input_indices.end()) {
      continue;
    }
    bool fatal_error_found = false;
    auto ret = CheckInputForPassThrough(gather_data_input_arg, current_node.InputDefs()[i], info, logger,
                                        fatal_error_found, input_has_dim_1_for_axis);
    if (fatal_error_found) {
      LOG_DEBUG_INFO(logger, "Skip for node " + current_node.Name() + " due to input check failure at index " +
                                 std::to_string(i));
      return false;
    } else if (ret.has_value()) {
      propagate_input_config[static_cast<int>(i)] = ret.value();
    }
  }

  // Make sure once Gather is moved before target node, all its outputs can be correctly be sliced.
  std::unordered_map<int, int> output_indices;
  for (size_t i = 0; i < current_node.OutputDefs().size(); ++i) {
    if (static_cast<int>(i) == current_node_output_index) {
      continue;
    }

    bool fatal_error_found = false;
    bool dim_1_for_axis_found = false;
    auto ret = CheckInputForPassThrough(gather_data_input_arg, current_node.OutputDefs()[i], info, logger,
                                        fatal_error_found, dim_1_for_axis_found);
    if (fatal_error_found) {
      LOG_DEBUG_INFO(logger, "Skip for node " + current_node.Name() + " due to output check failure at index " +
                                 std::to_string(i));
      return false;
    } else if (ret.has_value()) {
      output_indices[static_cast<int>(i)] = ret.value();
    }
  }
  bool output_check_success = output_indices.size() == current_node.OutputDefs().size() - 1;

  return output_check_success;
}

bool ReductionOpPassThroughActor::PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                                           const std::vector<int>& allowed_input_indices,
                                           const logging::Logger& logger,
                                           std::unordered_map<int, int>& propagate_input_config,
                                           bool& input_has_dim_1_for_axis) {
  auto axis = static_cast<int64_t>(current_node.GetAttributes().at("axis").i());
  axis = axis < 0 ? axis + current_node.InputDefs()[0]->Shape()->dim_size() : axis;

  // Make sure layernorm/softmax's reduction happens after the axis we want to slice.
  if (axis <= info.non_negative_axis) {
    return false;
  }

  return SimplePassThroughActor::PreCheck(graph, current_node, info, allowed_input_indices, logger,
                                          propagate_input_config, input_has_dim_1_for_axis);
}
bool ReshapePassThroughActor::PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                                       const std::vector<int>& /*allowed_input_indices*/,
                                       const logging::Logger& logger,
                                       std::unordered_map<int, int>& propagate_input_config,
                                       bool& /*input_has_dim_1_for_axis*/) {
  auto data_input_shape = current_node.InputDefs()[0]->Shape();
  auto shape_input_shape = current_node.InputDefs()[1]->Shape();
  auto output_shape = current_node.OutputDefs()[0]->Shape();
  if (data_input_shape == nullptr || shape_input_shape == nullptr || shape_input_shape->dim_size() != 1 ||
      output_shape == nullptr) {
    LOG_DEBUG_INFO(logger, "Reshape input/output node arg shape is not valid.");
    return false;
  }

  if (!graph_utils::IsConstantInitializer(graph, current_node.InputDefs()[1]->Name())) {
    LOG_DEBUG_INFO(logger, "Skip handle the Reshape, because the new shape is not constant.");
    return false;
  }

  propagate_input_config.clear();

  InlinedVector<int64_t> new_shape_const_values;
  optimizer_utils::AppendTensorFromInitializer(graph, *current_node.InputDefs()[1], new_shape_const_values, true);
  // Only below two cases are supported for easier updating shape data after propagate slice ops.
  // 1). If the shape data on slicing axis is zero (e.g. remain the same after slicing), we support it.
  // 2). Or if the sliced dim value is a constant, we also support it, and can update the shape data directly.
  // For other cases, it is feasible to support but we don't support for now.
  if (new_shape_const_values[info.non_negative_axis] == 0 || info.output_dim_on_axis.has_dim_value()) {
    auto in_dims = data_input_shape->dim();
    auto out_dims = output_shape->dim();
    int in_rank = in_dims.size();
    int out_rank = out_dims.size();

    int reshape_input_axis = -1;
    // Match from left to right.
    for (int i = 0; i < std::min(in_rank, out_rank); ++i) {
      bool dim_value_eq = in_dims[i].has_dim_value() && out_dims[i].has_dim_value() &&
                          in_dims[i].dim_value() == out_dims[i].dim_value();
      bool dim_param_eq = in_dims[i].has_dim_param() && out_dims[i].has_dim_param() &&
                          in_dims[i].dim_param() == out_dims[i].dim_param();
      if (dim_value_eq || dim_param_eq) {
        if (i == info.non_negative_axis) {
          reshape_input_axis = i;
          break;
        }
        continue;
      }
    }

    if (reshape_input_axis == -1) {
      // Match from right to left.
      for (int i = 0; i < std::min(in_rank, out_rank); ++i) {
        int in_index = in_rank - 1 - i;
        int out_index = out_rank - 1 - i;
        bool dim_value_eq = in_dims[in_index].has_dim_value() && out_dims[out_index].has_dim_value() &&
                            in_dims[in_index].dim_value() == out_dims[out_index].dim_value();
        bool dim_param_eq = in_dims[in_index].has_dim_param() && out_dims[out_index].has_dim_param() &&
                            in_dims[in_index].dim_param() == out_dims[out_index].dim_param();
        if (dim_value_eq || dim_param_eq) {
          if (out_index == info.non_negative_axis) {
            reshape_input_axis = in_index;
            break;
          }
          continue;
        }
      }
    }

    if (reshape_input_axis == -1) {
      LOG_DEBUG_INFO(logger, "Cannot find Reshape's input axis for Gather.");
      return false;
    }

    propagate_input_config[0] = reshape_input_axis;
    return true;
  }

  return false;
}

bool ReshapePassThroughActor::PostProcess(Graph& graph, Node& current_node, int /*current_node_output_index*/,
                                          int slice_axis, bool is_slice_scalar, bool /*input_has_dim_1_for_axis*/,
                                          const ONNX_NAMESPACE::TensorShapeProto_Dimension& output_dim_on_axis,
                                          const std::string& /*entry_node_name*/,
                                          const std::unordered_map<int, SliceInfo>& /*new_gather_infos*/,
                                          const logging::Logger& logger) {
  LOG_DEBUG_INFO(logger, "ReshapePostProcess for Node " + current_node.Name() + "(" + current_node.OpType() + ")");
  InlinedVector<int64_t> new_shape_const_values;
  optimizer_utils::AppendTensorFromInitializer(graph, *current_node.InputDefs()[1], new_shape_const_values, true);

  auto create_new_initializer_from_vector = [&graph](NodeArg* arg_to_be_replaced,
                                                     const InlinedVector<int64_t>& new_values) -> NodeArg* {
    // Create new TensorProto.
    ONNX_NAMESPACE::TensorProto constant_tensor_proto;
    constant_tensor_proto.set_name(graph.GenerateNodeArgName(arg_to_be_replaced->Name()));
    constant_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    auto length = new_values.size();
    constant_tensor_proto.add_dims(length);
    constant_tensor_proto.set_raw_data(new_values.data(), length * sizeof(int64_t));

    // Add initializer into Graph.
    NodeArg* new_shape_arg = &graph_utils::AddInitializer(graph, constant_tensor_proto);
    // Update the output arg shape.
    ONNX_NAMESPACE::TensorShapeProto new_shape;
    new_shape.add_dim()->set_dim_value(length);
    new_shape_arg->SetShape(new_shape);

    return new_shape_arg;
  };

  // If the shape constant on slice_axis is 0, then it keeps the original dim of input.
  // If it is scalar slice, then we just remove that dim. Otherwise, we don't need to update the dim value.
  if (new_shape_const_values[slice_axis] == 0) {
    if (is_slice_scalar) {
      LOG_DEBUG_INFO(logger, "Removing axis " + std::to_string(slice_axis) + " from shape tensor.");
      NodeArg* arg_to_be_replaced = current_node.MutableInputDefs()[1];
      InlinedVector<int64_t> new_values;
      for (int i = 0; i < static_cast<int>(new_shape_const_values.size()); ++i) {
        if (i != slice_axis) {
          new_values.push_back(new_shape_const_values[i]);
        }
      }
      auto new_shape_arg = create_new_initializer_from_vector(arg_to_be_replaced, new_values);
      graph_utils::ReplaceNodeInput(current_node, 1, *new_shape_arg);
    } else {
      LOG_DEBUG_INFO(logger, "Reshape's shape has 0 specified for aixs: " + std::to_string(slice_axis) +
                                 ", not need update.");
    }
    return true;
  }

  // If it selected shape is dim value, we can update the shape tensor directory.
  if (output_dim_on_axis.has_dim_value()) {
    new_shape_const_values[slice_axis] = output_dim_on_axis.dim_value();
    auto new_shape_arg = create_new_initializer_from_vector(current_node.MutableInputDefs()[1], new_shape_const_values);
    graph_utils::ReplaceNodeInput(current_node, 1, *new_shape_arg);
    return true;
  }

  ORT_THROW("Fail to update shape data in ReshapePassThroughActor::PostProcess, but this should not be called.");
}

bool TransposePassThroughActor::PreCheck(const Graph& /*graph*/, const Node& current_node, const SliceInfo& info,
                                         const std::vector<int>& /*allowed_input_indices*/,
                                         const logging::Logger& logger,
                                         std::unordered_map<int, int>& propagate_input_config,
                                         bool& input_has_dim_1_for_axis) {
  InlinedVector<int64_t> perm;
  if (!graph_utils::GetRepeatedNodeAttributeValues(current_node, "perm", perm)) {
    LOG_DEBUG_INFO(logger, "perm attribute is not set for node " + current_node.Name());
    return false;
  }
  propagate_input_config.clear();
  propagate_input_config[0] = static_cast<int>(perm[info.non_negative_axis]);
  input_has_dim_1_for_axis = false;
  return true;
}

bool TransposePassThroughActor::PostProcess(Graph& graph, Node& current_node, int current_node_output_index,
                                            int slice_axis, bool is_slice_scalar, bool /*input_has_dim_1_for_axis*/,
                                            const ONNX_NAMESPACE::TensorShapeProto_Dimension& /*output_dim_on_axis*/,
                                            const std::string& entry_node_name,
                                            const std::unordered_map<int, SliceInfo>& new_gather_infos,
                                            const logging::Logger& logger) {
  LOG_DEBUG_INFO(logger, "Enter TransposePassThroughActor::PostProcess for Node " + current_node.Name() + "(" +
                             current_node.OpType() + ")");

  // We need keep the original dimension to align with original perm.
  if (is_slice_scalar) {
    AdaptInputAndOutputForScalarSlice(graph, current_node, current_node_output_index, slice_axis,
                                      entry_node_name, new_gather_infos, logger);
  }
  return true;
}

bool MatMulPassThroughActor::PreCheck(const Graph& /*graph*/, const Node& current_node, const SliceInfo& info,
                                      const std::vector<int>& allowed_input_indices,
                                      const logging::Logger& logger,
                                      std::unordered_map<int, int>& propagate_input_config,
                                      bool& input_has_dim_1_for_axis) {
  LOG_DEBUG_INFO(logger, "Enter MatMulPassThroughActor::PreCheck for node " + current_node.Name());
  auto lhs_rank = current_node.InputDefs()[0]->Shape()->dim_size();
  auto rhs_rank = current_node.InputDefs()[1]->Shape()->dim_size();

  if (!(lhs_rank >= 2 && rhs_rank >= 2)) {
    LOG_DEBUG_INFO(logger, "MatMul input rank lower than 2, skip.");
    return false;
  }

  propagate_input_config.clear();
  if (info.non_negative_axis == info.input_rank - 1) {
    propagate_input_config[1] = rhs_rank - 1;
    return true;
  } else if (info.non_negative_axis == info.input_rank - 2) {
    propagate_input_config[0] = lhs_rank - 2;
    return true;
  }

  int target_node_output_index = optimizer_utils::IndexOfNodeOutput(current_node, *info.node_ptr->InputDefs()[0]);
  const NodeArg* gather_data_input_arg = current_node.OutputDefs()[target_node_output_index];

  input_has_dim_1_for_axis = false;
  for (size_t i = 0; i < current_node.InputDefs().size(); ++i) {
    if (allowed_input_indices.size() > 0 &&
        std::find(allowed_input_indices.begin(), allowed_input_indices.end(), i) == allowed_input_indices.end()) {
      continue;
    }
    bool fatal_error_found = false;
    auto ret = CheckInputForPassThrough(gather_data_input_arg, current_node.InputDefs()[i], info, logger,
                                        fatal_error_found, input_has_dim_1_for_axis);
    if (fatal_error_found) {
      LOG_DEBUG_INFO(logger, "Skip for node " + current_node.Name() + " due to input check failure at index " +
                                 std::to_string(i));
      return false;
    } else if (ret.has_value()) {
      LOG_DEBUG_INFO(logger, "Add new input candidate for node " + current_node.Name() + " at index " +
                                 std::to_string(i) + " with axis " + std::to_string(ret.value()));
      propagate_input_config[static_cast<int>(i)] = ret.value();
    }
  }

  return propagate_input_config.size() > 0;
}

bool MatMulPassThroughActor::PostProcess(Graph& graph, Node& current_node, int current_node_output_index,
                                         int slice_axis, bool is_slice_scalar, bool /*input_has_dim_1_for_axis*/,
                                         const ONNX_NAMESPACE::TensorShapeProto_Dimension& /*output_dim_on_axis*/,
                                         const std::string& entry_node_name,
                                         const std::unordered_map<int, SliceInfo>& new_gather_infos,
                                         const logging::Logger& logger) {
  LOG_DEBUG_INFO(logger, "Enter MatMulPassThroughActor::PostProcess for Node " + current_node.Name() + "(" +
                             current_node.OpType() + ")");

  // We need keep the original dimension to avoid the matmul inputs cannot be compatible to compute.
  if (is_slice_scalar) {
    AdaptInputAndOutputForScalarSlice(graph, current_node, current_node_output_index, slice_axis,
                                      entry_node_name, new_gather_infos, logger);
  }
  return true;
}

}  // namespace onnxruntime::optimizer::compute_optimizer

#endif
