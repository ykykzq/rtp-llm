#include <cstring>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

TEST(QueryConverterLogprobsTest, TranslatesGenerateConfig) {
    GenerateInputPB input;
    input.add_token_ids(1);
    auto* config = input.mutable_generate_config();
    config->set_return_logprobs(true);
    config->set_top_logprobs(2);

    auto generate_input = QueryConverter::transQuery(&input);

    ASSERT_NE(generate_input, nullptr);
    ASSERT_NE(generate_input->generate_config, nullptr);
    EXPECT_TRUE(generate_input->generate_config->return_logprobs);
    EXPECT_EQ(generate_input->generate_config->top_logprobs, 2);
}

TEST(QueryConverterLogprobsTest, TranslatesCompactOutputTensors) {
    GenerateOutput output;
    output.output_ids = torch::tensor({{1, 2, 3}}, torch::kInt32);
    output.finished   = true;
    output.token_logprobs.emplace(torch::tensor({-0.1f, -0.2f, -0.3f}, torch::kFloat32));
    output.top_logprob_token_ids.emplace(torch::tensor({{10, 11}, {12, 13}, {14, 15}}, torch::kInt32));
    output.top_logprobs.emplace(torch::tensor({{-1.0f, -2.0f}, {-1.1f, -2.1f}, {-1.2f, -2.2f}}, torch::kFloat32));

    GenerateOutputs outputs;
    outputs.request_id = 1;
    outputs.generate_outputs.push_back(std::move(output));

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs, false, "", 0);

    const auto& flatten_output = outputs_pb.flatten_output();

    ASSERT_TRUE(flatten_output.has_token_logprobs());
    const auto& token_logprobs = flatten_output.token_logprobs();
    EXPECT_EQ(token_logprobs.data_type(), TensorPB::FP32);
    ASSERT_EQ(token_logprobs.shape_size(), 2);
    EXPECT_EQ(token_logprobs.shape(0), 1);
    EXPECT_EQ(token_logprobs.shape(1), 3);
    std::vector<float> token_logprob_values(3);
    std::memcpy(token_logprob_values.data(), token_logprobs.fp32_data().data(), token_logprobs.fp32_data().size());
    EXPECT_FLOAT_EQ(token_logprob_values[0], -0.1f);
    EXPECT_FLOAT_EQ(token_logprob_values[2], -0.3f);

    ASSERT_TRUE(flatten_output.has_top_logprob_token_ids());
    const auto& top_token_ids = flatten_output.top_logprob_token_ids();
    EXPECT_EQ(top_token_ids.data_type(), TensorPB::INT32);
    ASSERT_EQ(top_token_ids.shape_size(), 3);
    EXPECT_EQ(top_token_ids.shape(0), 1);
    EXPECT_EQ(top_token_ids.shape(1), 3);
    EXPECT_EQ(top_token_ids.shape(2), 2);
    std::vector<int32_t> top_token_id_values(6);
    std::memcpy(top_token_id_values.data(), top_token_ids.int32_data().data(), top_token_ids.int32_data().size());
    EXPECT_EQ(top_token_id_values[0], 10);
    EXPECT_EQ(top_token_id_values[5], 15);

    ASSERT_TRUE(flatten_output.has_top_logprobs());
    const auto& top_logprobs = flatten_output.top_logprobs();
    EXPECT_EQ(top_logprobs.data_type(), TensorPB::FP32);
    ASSERT_EQ(top_logprobs.shape_size(), 3);
    EXPECT_EQ(top_logprobs.shape(0), 1);
    EXPECT_EQ(top_logprobs.shape(1), 3);
    EXPECT_EQ(top_logprobs.shape(2), 2);
    std::vector<float> top_logprob_values(6);
    std::memcpy(top_logprob_values.data(), top_logprobs.fp32_data().data(), top_logprobs.fp32_data().size());
    EXPECT_FLOAT_EQ(top_logprob_values[0], -1.0f);
    EXPECT_FLOAT_EQ(top_logprob_values[5], -2.2f);
}

TEST(QueryConverterLogprobsTest, OmitsLogprobSubmessagesWhenDisabled) {
    GenerateOutput output;
    output.output_ids = torch::tensor({{1, 2}}, torch::kInt32);
    output.finished   = true;

    GenerateOutputs outputs;
    outputs.request_id = 2;
    outputs.generate_outputs.push_back(std::move(output));

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs, false, "", 0);

    ASSERT_TRUE(outputs_pb.has_flatten_output());
    const auto& flatten_output = outputs_pb.flatten_output();
    EXPECT_FALSE(flatten_output.has_token_logprobs());
    EXPECT_FALSE(flatten_output.has_top_logprob_token_ids());
    EXPECT_FALSE(flatten_output.has_top_logprobs());
}

TEST(QueryConverterLogprobsTest, PreservesZeroTopKLogprobSubmessages) {
    GenerateInputPB input;
    input.add_token_ids(1);
    auto* config = input.mutable_generate_config();
    config->set_return_logprobs(true);
    config->set_top_logprobs(0);

    auto generate_input = QueryConverter::transQuery(&input);
    ASSERT_NE(generate_input, nullptr);
    ASSERT_NE(generate_input->generate_config, nullptr);
    EXPECT_TRUE(generate_input->generate_config->return_logprobs);
    EXPECT_EQ(generate_input->generate_config->top_logprobs, 0);

    GenerateOutput output;
    output.output_ids = torch::tensor({{1, 2}}, torch::kInt32);
    output.finished   = true;
    output.token_logprobs.emplace(torch::tensor({-0.1f, -0.2f}, torch::kFloat32));
    output.top_logprob_token_ids.emplace(torch::empty({2, 0}, torch::kInt32));
    output.top_logprobs.emplace(torch::empty({2, 0}, torch::kFloat32));

    GenerateOutputs outputs;
    outputs.request_id = 3;
    outputs.generate_outputs.push_back(std::move(output));

    GenerateOutputsPB outputs_pb;
    QueryConverter::transResponse(&outputs_pb, &outputs, false, "", 0);

    const auto& flatten_output = outputs_pb.flatten_output();
    ASSERT_TRUE(flatten_output.has_token_logprobs());
    EXPECT_EQ(flatten_output.token_logprobs().data_type(), TensorPB::FP32);
    ASSERT_EQ(flatten_output.token_logprobs().shape_size(), 2);
    EXPECT_EQ(flatten_output.token_logprobs().shape(0), 1);
    EXPECT_EQ(flatten_output.token_logprobs().shape(1), 2);

    ASSERT_TRUE(flatten_output.has_top_logprob_token_ids());
    EXPECT_EQ(flatten_output.top_logprob_token_ids().data_type(), TensorPB::INT32);
    ASSERT_EQ(flatten_output.top_logprob_token_ids().shape_size(), 3);
    EXPECT_EQ(flatten_output.top_logprob_token_ids().shape(0), 1);
    EXPECT_EQ(flatten_output.top_logprob_token_ids().shape(1), 2);
    EXPECT_EQ(flatten_output.top_logprob_token_ids().shape(2), 0);
    EXPECT_TRUE(flatten_output.top_logprob_token_ids().int32_data().empty());

    ASSERT_TRUE(flatten_output.has_top_logprobs());
    EXPECT_EQ(flatten_output.top_logprobs().data_type(), TensorPB::FP32);
    ASSERT_EQ(flatten_output.top_logprobs().shape_size(), 3);
    EXPECT_EQ(flatten_output.top_logprobs().shape(0), 1);
    EXPECT_EQ(flatten_output.top_logprobs().shape(1), 2);
    EXPECT_EQ(flatten_output.top_logprobs().shape(2), 0);
    EXPECT_TRUE(flatten_output.top_logprobs().fp32_data().empty());
}

}  // namespace rtp_llm
