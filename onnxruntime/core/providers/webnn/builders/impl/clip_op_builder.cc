// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class ClipOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType device_type, const logging::Logger& logger) const override;
};

// Add operator related.

void ClipOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Both min and max values will be injected into the layer, no need to add to the model.
  if (node.SinceVersion() >= 11) {
    if (node.InputDefs().size() > 1)
      model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());

    if (node.InputDefs().size() > 2)
      model_builder.AddInitializerToSkip(node.InputDefs()[2]->Name());
  }
}

Status ClipOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                            const Node& node,
                                            const logging::Logger& logger) const {
  const auto& input_name = node.InputDefs()[0]->Name();
  const auto& output_name = node.OutputDefs()[0]->Name();
  emscripten::val options = emscripten::val::object();
  float minValue, maxValue;
  ORT_RETURN_IF_NOT(GetClipMinMax(model_builder.GetGraphViewer(), node, minValue, maxValue, logger),
                    "GetClipMinMax failed");
  options.set("minValue", minValue);
  options.set("maxValue", maxValue);
  options.set("label", node.Name());
  emscripten::val input = model_builder.GetOperand(input_name);
  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("clamp", input, options);

  model_builder.AddOperand(output_name, std::move(output));
  return Status::OK();
}

// Operator support related.

bool ClipOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                      const Node& node,
                                      const WebnnDeviceType device_type,
                                      const logging::Logger& logger) const {
  // TODO: Update IsOpSupportedImpl to pass GraphViewer instead of InitializedTensorSet so the implementations
  // can ensure initializers are constant. See #19401 for details of how this update was made to the NNAPI EP.
  // GetClipMinMax(graph_viewer, node, minValue, maxValue, logger)
  float min, max;
  if (GetClipMinMax(initializers, node, min, max, logger)) {
    // WebNN CPU backend only supports 3 specific ranges: [0.0, infinity], [-1.0, 1.0], [0.0, 6.0].
    // TODO: Remove this workaround once the associated issue is resolved in Chromium:
    // https://issues.chromium.org/issues/326156496.
    if (device_type == WebnnDeviceType::CPU) {
      if ((min == 0.0f && max == std::numeric_limits<float>::infinity()) ||
          (min == -1.0f && max == 1.0f) ||
          (min == 0.0f && max == 6.0f)) {
        return true;
      } else {
        LOGS(logger, VERBOSE) << "Clip min and max values ("
                              << min << ", "
                              << max << ") are not supported for WebNN CPU backend";
        return false;
      }
    }

    return true;
  } else {
    return false;
  };
}

void CreateClipOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ClipOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
