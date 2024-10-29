// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/multihead_attention_helper.h"
#include "contrib_ops/webgpu/bert/attention.h"
#include "contrib_ops/webgpu/bert/multihead_attention.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
using namespace onnxruntime::webgpu;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::contrib::multihead_attention_helper;

namespace onnxruntime {
namespace contrib {
namespace webgpu {

Status TransferBSDToBNSHProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("qkv_input", ShaderUsage::UseUniform);
  const auto& qkv_output = shader.AddOutput("qkv_output", ShaderUsage::UseUniform | ShaderUsage::UseOffsetToIndices);

  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.data_size")
                            << "let output_indices = " << qkv_output.OffsetToIndices("global_idx") << ";\n"
                            << "let input_offset_idx = output_indices[0] * uniforms.batch_offset + output_indices[1] *"
                            << " uniforms.head_offset + output_indices[2] * uniforms.sequence_offset + output_indices[3];\n";
  if (has_bias_) {
    shader.MainFunctionBody() << "let bias_offset_idx = (input_offset_idx % uniforms.sequence_offset) + uniforms.bias_offset;\n";
  }
  shader.MainFunctionBody() << "qkv_output[global_idx] = qkv_input[input_offset_idx]";
  if (has_bias_) {
    shader.MainFunctionBody() << " + bias[bias_offset_idx];\n";
  } else {
    shader.MainFunctionBody() << ";\n";
  }

  return Status::OK();
}

Status TransferBSDToBNSH(onnxruntime::webgpu::ComputeContext& context, int num_heads, int sequence_length,
                         int head_size, const Tensor* input_tensor, const Tensor* bias, int bias_offset, Tensor* output_tensor) {
  assert(input_tensor->Shape().GetDims().size() == 3);
  assert(output_tensor->Shape().GetDims().size() == 4);

  uint32_t data_size = SafeInt<uint32_t>(output_tensor->Shape().Size());
  const int batch_offset = num_heads * sequence_length * head_size;
  const int sequence_offset = num_heads * head_size;
  const int head_offset = head_size;
  bool has_bias = bias != nullptr;

  TransferBSDToBNSHProgram program{has_bias};
  program.AddInputs({{input_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .AddOutputs({{output_tensor, ProgramTensorMetadataDependency::TypeAndRank}})
      .SetDispatchGroupSize((data_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({{data_size},
                            {static_cast<uint32_t>(batch_offset)},
                            {static_cast<uint32_t>(sequence_offset)},
                            {static_cast<uint32_t>(head_offset)},
                            {static_cast<uint32_t>(bias_offset)}});

  if (has_bias) {
    program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank});
  }

  return context.RunProgram(program);
};

std::string InitVarStub(const Tensor* seqlens_k, const Tensor* total_seqlen_tensor, bool init_past_sequence_length) {
  std::ostringstream ss;
  if (seqlens_k != nullptr && total_seqlen_tensor != nullptr) {
    ss << "let total_sequence_length_input = u32(total_seqlen_tensor[0]);\n";
    ss << "let present_sequence_length = max(total_sequence_length_input, uniforms.past_sequence_length);\n";
    ss << "let is_subsequent_prompt: bool = sequence_length > 1 && sequence_length != total_sequence_length_input;\n";
    ss << "let is_first_prompt: bool = is_subsequent_prompt == false && sequence_length == total_sequence_length_input;\n";
    ss << "total_sequence_length = u32(seqlens_k[batch_idx]) + 1;\n";
    ss << "var past_sequence_length: u32 = 0;\n";
    ss << "if (is_first_prompt == false) {\n";
    ss << "  past_sequence_length = total_sequence_length - sequence_length;\n";
    ss << "}\n";
  } else {
    if (init_past_sequence_length) {
      ss << "let past_sequence_length = uniforms.past_sequence_length;\n";
    }
    ss << "let present_sequence_length = total_sequence_length;\n";
  }
  return ss.str();
}

Status AttentionProbsProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("q", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  shader.AddInput("key", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  if (feed_past_key_) {
    shader.AddInput("past_key", ShaderUsage::UseUniform);
  }
  if (has_attention_bias_) {
    shader.AddInput("attention_bias", ShaderUsage::UseUniform);
  }
  if (seqlen_k_ != nullptr) {
    shader.AddInput("seqlens_k", ShaderUsage::UseUniform);
  }
  if (total_seqlen_tensor_ != nullptr) {
    shader.AddInput("total_seqlen_tensor", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  if (has_present_key_) {
    shader.AddOutput("present_key", ShaderUsage::UseUniform);
  }

  shader.AdditionalImplementation() << "var<workgroup> tileQ: array<q_value_t, " << tile_size_ * tile_size_ << ">;\n"
                                    << "var<workgroup> tileK: array<key_value_t, " << tile_size_ * tile_size_ << ">;\n"
                                    << "alias f32_val_t = " << (components_ == 4 ? "vec4<f32>" : (components_ == 2 ? "vec2<f32>" : "f32")) << ";\n";
  shader.MainFunctionBody() << "// x holds the N and y holds the M\n"
                               "let head_idx = workgroup_id.z % uniforms.num_heads;\n"
                               "let kv_head_idx = head_idx / uniforms.n_reps;\n"
                               "let kv_num_heads = uniforms.num_heads / uniforms.n_reps;\n"
                               "let batch_idx = workgroup_id.z / uniforms.num_heads;\n"
                               "let m = workgroup_id.y * TILE_SIZE;\n"
                               "let n = workgroup_id.x * TILE_SIZE;\n"
                               "let sequence_length = uniforms.M;\n"
                               "var total_sequence_length = uniforms.N;\n"
                               "let abs_kv_head_idx = batch_idx * kv_num_heads + kv_head_idx;\n"
                               "let qOffset = workgroup_id.z * uniforms.M * uniforms.K + m * uniforms.K;\n";

  shader.MainFunctionBody() << InitVarStub(seqlen_k_, total_seqlen_tensor_, true);

  if (feed_past_key_ && has_present_key_) {
    shader.MainFunctionBody() << "let pastKeyOffset = abs_kv_head_idx * uniforms.past_sequence_length * uniforms.K;\n";
  }
  shader.MainFunctionBody() << "let kOffset = abs_kv_head_idx * uniforms.kv_sequence_length * uniforms.K;\n";
  if (has_present_key_) {
    shader.MainFunctionBody() << "let presentKeyOffset = abs_kv_head_idx * uniforms.N * uniforms.K;\n";
  }

  shader.MainFunctionBody() << "var value = f32_val_t(0);\n"
                               "for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {\n"
                               "  if (global_id.y < uniforms.M && w + local_id.x < uniforms.K) {\n"
                               "    tileQ[TILE_SIZE * local_id.y + local_id.x] = q[qOffset + local_id.y * uniforms.K + w + local_id.x];\n"
                               "  }\n"
                               "  if (n + local_id.y < uniforms.N && w + local_id.x < uniforms.K) {\n"
                               "    var idx = TILE_SIZE * local_id.y + local_id.x;\n";

  if (feed_past_key_ && has_present_key_) {
    shader.MainFunctionBody() << "    if (n + local_id.y < past_sequence_length) {\n"
                                 "      tileK[idx] = past_key[pastKeyOffset + (n + local_id.y) * uniforms.K + w + local_id.x];\n"
                                 "    } else  if (n + local_id.y - past_sequence_length < uniforms.kv_sequence_length) {\n"
                                 "      tileK[idx] = key[kOffset + (n + local_id.y - past_sequence_length) * uniforms.K + w + local_id.x];\n"
                                 "    }\n";
  } else {
    shader.MainFunctionBody() << "    if (n + local_id.y < uniforms.kv_sequence_length) {\n"
                                 "      tileK[idx] = key[kOffset + (n + local_id.y) * uniforms.K + w + local_id.x];\n"
                                 "    }\n";
  }

  if (has_present_key_) {
    shader.MainFunctionBody() << "    if (n + local_id.y < present_sequence_length) {\n"
                              << "      present_key[presentKeyOffset + (n + local_id.y) * uniforms.K + w + local_id.x] = tileK[idx];\n"
                              << "    }\n";
  }

  shader.MainFunctionBody() << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "  for (var k: u32 = 0u; k < TILE_SIZE && w+k < uniforms.K; k++) {\n"
                            << "    value += f32_val_t(tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * local_id.x + k]);\n"
                            << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "}\n";

  shader.MainFunctionBody() << "if (global_id.y < uniforms.M && global_id.x < total_sequence_length) {\n"
                            << "  let headOffset = workgroup_id.z * uniforms.M * uniforms.N;\n"
                            << "  let outputIdx = headOffset + global_id.y * uniforms.N + global_id.x;\n"
                            << "  var sum: f32 = " << (components_ == 4 ? "value.x + value.y + value.z + value.w" : (components_ == 2 ? "value.x + value.y" : "value")) << ";\n";

  shader.MainFunctionBody() << "  output[outputIdx] = output_value_t(sum * uniforms.alpha)";
  if (has_attention_bias_) {
    shader.MainFunctionBody() << " + attention_bias[outputIdx]";
  }
  shader.MainFunctionBody() << ";\n"
                            << "}\n";

  return Status::OK();
}

Status ComputeAttentionProbs(onnxruntime::webgpu::ComputeContext& context, int output_count, const Tensor* Q,
                             const Tensor* K, const Tensor* past_key, const Tensor* attention_bias, Tensor* probs, Tensor* present_key,
                             WebgpuAttentionParameters& parameters, int past_sequence_length, int total_sequence_length,
                             const Tensor* seqlen_k, const Tensor* total_seqlen_tensor) {
  const float alpha = parameters.scale_ == 0.0f ? 1.f / sqrt(static_cast<float>(parameters.head_size_))
                                                : parameters.scale_;

  const bool feed_past_key = present_key != nullptr && past_key != nullptr && past_key->SizeInBytes() > 0;
  const bool has_present_key = output_count > 1 && past_key;
  const bool has_attention_bias = attention_bias != nullptr;
  const int tile_size = 12;
  const int components = parameters.head_size_ % 4 == 0 ? 4 : (parameters.head_size_ % 2 == 0 ? 2 : 1);

  AttentionProbsProgram program{"AttentionProbs", feed_past_key, has_present_key, has_attention_bias, tile_size,
                                components, seqlen_k, total_seqlen_tensor};
  program.AddInputs({{Q, ProgramTensorMetadataDependency::TypeAndRank, components},
                     {K, ProgramTensorMetadataDependency::TypeAndRank, components}});
  if (feed_past_key) {
    program.AddInput({past_key, ProgramTensorMetadataDependency::TypeAndRank, components});
  }
  if (has_attention_bias) {
    program.AddInput({attention_bias, ProgramTensorMetadataDependency::TypeAndRank});
  }
  if (seqlen_k != nullptr && total_seqlen_tensor != nullptr) {
    program.AddInputs({{seqlen_k, ProgramTensorMetadataDependency::TypeAndRank},
                       {total_seqlen_tensor, ProgramTensorMetadataDependency::TypeAndRank}});
  }
  program.AddOutputs({{probs, ProgramTensorMetadataDependency::Rank}});
  if (has_present_key) {
    program.AddOutput({present_key, ProgramTensorMetadataDependency::Rank, components});
  }

  const uint32_t vectorized_head_size = parameters.head_size_ / components;
  program.SetDispatchGroupSize((total_sequence_length + tile_size - 1) / tile_size,
                               (parameters.sequence_length_ + tile_size - 1) / tile_size,
                               parameters.batch_size_ * parameters.num_heads_)
      .SetWorkgroupSize(tile_size, tile_size)
      .CacheHint(std::to_string(tile_size))
      .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(vectorized_head_size)},
                            {static_cast<uint32_t>(total_sequence_length)},
                            {static_cast<uint32_t>(parameters.num_heads_)},
                            {static_cast<uint32_t>(parameters.head_size_)},
                            {static_cast<float>(alpha)},
                            {static_cast<uint32_t>(past_sequence_length)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length_)},
                            {static_cast<uint32_t>(parameters.n_reps)}})
      .SetOverridableConstants({{static_cast<uint32_t>(tile_size)}});

  return context.RunProgram(program);
}

Status InPlaceSoftmaxProgram::GenerateShaderCode(ShaderHelper& shader) const {
  if (seqlen_k_) {
    shader.AddInput("seqlens_k", ShaderUsage::UseUniform);
  }
  if (total_seqlen_tensor_) {
    shader.AddInput("total_seqlen_tensor", ShaderUsage::UseUniform);
  }
  shader.AddOutput("x", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AdditionalImplementation() << "var<workgroup> thread_max: array<f32, " << work_group_size_ << ">;\n"
                                    << "var<workgroup> thread_sum: array<f32, " << work_group_size_ << ">;\n"
                                    << "alias f32_val_t = " << (components_ == 4 ? "vec4<f32>" : (components_ == 2 ? "vec2<f32>" : "f32")) << ";\n";
  shader.MainFunctionBody() << "let batch_idx = workgroup_id.z / uniforms.num_heads;\n"
                            << "let head_idx = workgroup_id.z % uniforms.num_heads;\n"
                            << "let sequence_length = uniforms.sequence_length;\n"
                            << "var total_sequence_length = uniforms.total_sequence_length;\n"
                            << InitVarStub(seqlen_k_, total_seqlen_tensor_, true).c_str()
                            << "let local_offset = local_idx * uniforms.elements_per_thread;\n"
                            << "let offset = (global_idx / " << work_group_size_ << ") * uniforms.total_sequence_length + local_offset;\n"
                            << "let seq_causal_length = " << (seqlen_k_ ? "past_sequence_length + workgroup_id.y + 1" : "total_sequence_length") << ";\n"
                            << "var thread_max_vector = f32_val_t(-3.402823e+38f);\n"
                            << "for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < seq_causal_length; i++) {\n"
                            << "  thread_max_vector = max(f32_val_t(x[offset + i]), thread_max_vector);\n"
                            << "}\n"
                            << "thread_max[local_idx] = " << (components_ == 4 ? "max(max(thread_max_vector.x, thread_max_vector.y), max(thread_max_vector.z, thread_max_vector.w))" : (components_ == 2 ? "max(thread_max_vector.x, thread_max_vector.y)" : "thread_max_vector")) << ";\n"
                            << "workgroupBarrier();\n"
                            << "var max_value =  f32(-3.402823e+38f);\n"
                            << "for (var i = 0u; i < " << work_group_size_ << "; i++) {\n"
                            << "  max_value = max(thread_max[i], max_value);\n"
                            << "}\n"
                            << "var sum_vector = f32_val_t(0);\n"
                            << "for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < seq_causal_length; i++) {\n"
                            << "  sum_vector += exp(f32_val_t(x[offset + i]) - max_value);\n"
                            << "}\n"
                            << "thread_sum[local_idx] = " << (components_ == 4 ? "sum_vector.x + sum_vector.y + sum_vector.z + sum_vector.w" : (components_ == 2 ? "sum_vector.x + sum_vector.y" : "sum_vector")) << ";\n"
                            << "workgroupBarrier();\n"
                            << "var sum: f32 = 0;\n"
                            << "for (var i = 0u; i < " << work_group_size_ << "; i++) {\n"
                            << "  sum += thread_sum[i]\n;"
                            << "}\n"
                            << "if (sum == 0) {\n"
                            << "  for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < seq_causal_length; i++) {\n"
                            << "    x[offset + i] = x_value_t(x_element_t(1.0)/x_element_t(seq_causal_length));\n"
                            << "  }\n"
                            << "} else {\n"
                            << "  for (var i: u32 = 0; i < uniforms.elements_per_thread && i + local_offset < seq_causal_length; i++) {\n"
                            << "    var f32input = f32_val_t(x[offset + i]);\n"
                            << "    x[offset + i] = x_value_t(exp(f32input - max_value) / sum);\n"
                            << "  }\n"
                            << "}\n";
  if (seqlen_k_) {
    shader.MainFunctionBody() << "for (var total_seq_id: u32 = seq_causal_length; total_seq_id + local_offset < uniforms.total_sequence_length; total_seq_id++) {\n"
                              << "   x[offset + total_seq_id] = x_value_t(x_element_t(0));\n"
                              << "}\n";
  }

  return Status::OK();
}

Status ComputeInPlaceSoftmax(onnxruntime::webgpu::ComputeContext& context, Tensor* probs, int32_t batch_size, int32_t num_heads, int32_t past_sequence_length, int32_t sequence_length, int32_t total_sequence_length,
                             const Tensor* seqlen_k, const Tensor* total_seqlen_tensor) {
  const int components = seqlen_k != nullptr ? 1 : (total_sequence_length % 4 == 0 ? 4 : (total_sequence_length % 2 == 0 ? 2 : 1));
  int work_group_size = 64;
  const int total_sequence_length_comp = total_sequence_length / components;
  if (total_sequence_length_comp < work_group_size) {
    work_group_size = 32;
  }
  const int elementsPerThread = (total_sequence_length_comp + work_group_size - 1) / work_group_size;

  InPlaceSoftmaxProgram program{"InPlaceSoftmax", work_group_size, components, seqlen_k, total_seqlen_tensor};
  if (seqlen_k != nullptr && total_seqlen_tensor != nullptr) {
    program.AddInputs({{seqlen_k, ProgramTensorMetadataDependency::TypeAndRank},
                       {total_seqlen_tensor, ProgramTensorMetadataDependency::TypeAndRank}});
  }
  program.AddOutputs({{probs, ProgramTensorMetadataDependency::TypeAndRank, components}})
      .SetDispatchGroupSize(batch_size * num_heads, sequence_length, total_sequence_length)
      .SetWorkgroupSize(work_group_size)
      .AddUniformVariables({{static_cast<uint32_t>(batch_size)},
                            {static_cast<uint32_t>(num_heads)},
                            {static_cast<uint32_t>(past_sequence_length)},
                            {static_cast<uint32_t>(sequence_length)},
                            {static_cast<uint32_t>(total_sequence_length)},
                            {static_cast<uint32_t>(elementsPerThread)}});

  return context.RunProgram(program);
}

Status VxAttentionScoreProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("probs", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AddInput("v", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias);
  if (feed_past_value_) {
    shader.AddInput("past_value", ShaderUsage::UseUniform);
  }
  if (seqlen_k_) {
    shader.AddInput("seqlens_k", ShaderUsage::UseUniform);
  }
  if (total_seqlen_tensor_) {
    shader.AddInput("total_seqlen_tensor", ShaderUsage::UseUniform);
  }
  shader.AddOutput("output", ShaderUsage::UseUniform);
  if (has_present_value_) {
    shader.AddOutput("present_value", ShaderUsage::UseUniform);
  }

  shader.AdditionalImplementation() << "var<workgroup> tileQ: array<probs_value_t, " << tile_size_ * tile_size_ << ">;\n"
                                    << "var<workgroup> tileK: array<v_value_t, " << tile_size_ * tile_size_ << ">;\n";
  shader.MainFunctionBody() << "let head_idx = workgroup_id.z % uniforms.num_heads;\n"
                            << "let batch_idx = workgroup_id.z / uniforms.num_heads;\n"
                            << "let kv_head_idx = head_idx / uniforms.n_reps;\n"
                            << "let kv_num_heads = uniforms.num_heads / uniforms.n_reps;\n"
                            << "let m = global_id.y;\n"
                            << "let n = global_id.x;\n"
                            << "let sequence_length = uniforms.M;\n"
                            << "var total_sequence_length = uniforms.K;\n"
                            << InitVarStub(seqlen_k_, total_seqlen_tensor_, true).c_str()
                            << "let abs_kv_head_idx = batch_idx * kv_num_heads + kv_head_idx;\n"
                            << "let vOffset = abs_kv_head_idx * uniforms.N * uniforms.kv_sequence_length + n;\n";

  if (feed_past_value_ && has_present_value_) {
    shader.MainFunctionBody() << "let pastValueOffset = abs_kv_head_idx * uniforms.N * uniforms.past_sequence_length + n\n";
  }

  if (has_present_value_) {
    shader.MainFunctionBody() << "let presentValueOffset = head_idx * uniforms.N * uniforms.K + n;\n";
  }

  shader.MainFunctionBody() << "var value = probs_element_t(0);\n"
                            << "for (var w: u32 = 0u; w < uniforms.K; w += TILE_SIZE) {\n"
                            << "  if (m < uniforms.M && w + local_id.x < uniforms.K) {\n"
                            << "    tileQ[TILE_SIZE * local_id.y + local_id.x] = probs[offsetA + w + local_id.x];\n"
                            << "  }\n"
                            << "  if (n < uniforms.N && w + local_id.y < uniforms.K) {\n"
                            << "    var idx = TILE_SIZE * local_id.y + local_id.x;\n";

  if (feed_past_value_ && has_present_value_) {
    shader.MainFunctionBody() << "    if (w + local_id.y < past_sequence_length) {\n"
                              << "      tileK[idx] = past_value[pastValueOffset + (w + local_id.y) * uniforms.N];\n"
                              << "    } else if (w + local_id.y - past_sequence_length < uniforms.kv_sequence_length) {\n"
                              << "      tileK[idx] = v[vOffset + (w + local_id.y - uniforms.past_sequence_length) * uniforms.N];\n"
                              << "    }\n";
  } else {
    shader.MainFunctionBody() << "    if (w + local_id.y < uniforms.kv_sequence_length) {\n"
                              << "      tileV[idx] = v[vOffset + (w + local_id.y) * uniforms.N];\n"
                              << "    }\n";
  }

  if (has_present_value_) {
    shader.MainFunctionBody() << "    if (w + local_id.y < present_sequence_length) {\n"
                              << "      present_value[presentValueOffset + (w + local_id.y) * uniforms.N] = tileV[idx];\n"
                              << "    }\n";
  }

  shader.MainFunctionBody() << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "  for (var k: u32 = 0u; k < TILE_SIZE && w+k < total_sequence_length; k++) {\n"
                            << "    value += tileQ[TILE_SIZE * local_id.y + k] * tileK[TILE_SIZE * k + local_id.x];\n"
                            << "  }\n"
                            << "  workgroupBarrier();\n"
                            << "}\n";

  shader.MainFunctionBody() << "// we need to transpose output from BNSH_v to BSND_v\n"
                            << "let batch_idx = workgroup_id.z / uniforms.num_heads;\n"
                            << "let currentBatchHeadNumber = workgroup_id.z % uniforms.num_heads;\n"
                            << "if (m < uniforms.M && n < uniforms.N) {\n"
                            << "  let outputIdx = batch_idx * uniforms.M * uniforms.v_hidden_size + "
                            << "  m * uniforms.v_hidden_size + head_idx * uniforms.N + n;\n"
                            << "  output[outputIdx] = value;\n"
                            << "}\n";

  return Status::OK();
}

Status ComputeVxAttentionScore(onnxruntime::webgpu::ComputeContext& context, int output_count,
                               const Tensor* probs,
                               const Tensor* V,
                               const Tensor* past_value,
                               Tensor* output,
                               Tensor* present_value,
                               WebgpuAttentionParameters& parameters,
                               int past_sequence_length,
                               int total_sequence_length,
                               const Tensor* seqlen_k,
                               const Tensor* total_seqlen_tensor) {
  const bool feed_past_value = present_value != nullptr && past_value != nullptr && past_value->SizeInBytes() > 0;
  const bool has_present_value = output_count > 1 && past_value != nullptr;
  const int tile_size = 12;

  VxAttentionScoreProgram program{"VxAttentionScore", feed_past_value, has_present_value, tile_size, seqlen_k, total_seqlen_tensor};
  program.AddInputs({{probs, ProgramTensorMetadataDependency::TypeAndRank},
                     {V, ProgramTensorMetadataDependency::TypeAndRank}});
  if (feed_past_value) {
    program.AddInput({past_value, ProgramTensorMetadataDependency::TypeAndRank});
  }
  if (seqlen_k != nullptr && total_seqlen_tensor != nullptr) {
    program.AddInputs({{seqlen_k, ProgramTensorMetadataDependency::TypeAndRank},
                       {total_seqlen_tensor, ProgramTensorMetadataDependency::TypeAndRank}});
  }
  program.AddOutputs({{output, ProgramTensorMetadataDependency::TypeAndRank}});
  if (has_present_value) {
    program.AddOutput({present_value, ProgramTensorMetadataDependency::TypeAndRank});
  }

  program.SetDispatchGroupSize((parameters.v_head_size_ + tile_size - 1) / tile_size,
                               (parameters.sequence_length_ + tile_size - 1) / tile_size,
                               parameters.batch_size_ * parameters.num_heads_)
      .SetWorkgroupSize(tile_size, tile_size)
      .AddUniformVariables({{static_cast<uint32_t>(parameters.sequence_length_)},
                            {static_cast<uint32_t>(total_sequence_length)},
                            {static_cast<uint32_t>(parameters.v_head_size_)},
                            {static_cast<uint32_t>(parameters.num_heads_)},
                            {static_cast<uint32_t>(parameters.head_size_)},
                            {static_cast<uint32_t>(parameters.v_hidden_size_)},
                            {static_cast<uint32_t>(past_sequence_length)},
                            {static_cast<uint32_t>(parameters.kv_sequence_length_)},
                            {static_cast<uint32_t>(parameters.n_reps)}})
      .SetOverridableConstants({{static_cast<uint32_t>(tile_size)}});
  ;

  return context.RunProgram(program);
}

Status ApplyAttention(const Tensor* Q, const Tensor* K, const Tensor* V, const Tensor* attention_bias,
                      const Tensor* past_key, const Tensor* past_value, Tensor* output, Tensor* present_key, Tensor* present_value,
                      WebgpuAttentionParameters& parameters, onnxruntime::webgpu::ComputeContext& context, const Tensor* seqlen_k, const Tensor* total_seqlen_tensor) {
  const int output_count = std::min({context.OutputCount(), 1 + (past_key != nullptr ? 1 : 0) + (past_value != nullptr ? 1 : 0)});
  const int past_sequence_length = output_count > 1 ? parameters.past_sequence_length_ : 0;
  const int total_sequence_length = past_sequence_length + parameters.kv_sequence_length_;

  const TensorShapeVector probs_dims({parameters.batch_size_, parameters.num_heads_,
                                      parameters.sequence_length_, total_sequence_length});
  const TensorShape probs_shape(probs_dims);
  Tensor probs = context.CreateGPUTensor(Q->DataType(), probs_shape);
  ORT_RETURN_IF_ERROR(ComputeAttentionProbs(context, output_count, Q, K, past_key, attention_bias, &probs, present_key,
                                            parameters, past_sequence_length, total_sequence_length, seqlen_k, total_seqlen_tensor));

  ORT_RETURN_IF_ERROR(ComputeInPlaceSoftmax(context, &probs,
                                            parameters.batch_size_, parameters.num_heads_, parameters.past_sequence_length_, parameters.sequence_length_, total_sequence_length, seqlen_k, total_seqlen_tensor));

  ORT_RETURN_IF_ERROR(ComputeVxAttentionScore(context, output_count, &probs, V, past_value, output, present_value,
                                              parameters, past_sequence_length, total_sequence_length, seqlen_k, total_seqlen_tensor));

  return Status::OK();
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime