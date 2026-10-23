#include <cstdint>
#include <iostream>
#include <vector>

#include "rtp_llm/cpp/utils/HashUtil.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    std::vector<int32_t> mutable_tokens{1, 42, 0, 151643, 27, 512};
    const std::vector<int32_t> tokens = mutable_tokens;
    for (const int64_t salt : {int64_t{0}, int64_t{7}, int64_t{4336115506469590333}}) {
        const auto expected = rtp_llm::hashInt64Array(
            salt, mutable_tokens.data(), mutable_tokens.data() + mutable_tokens.size());
        assert(rtp_llm::hashInt64Array(salt, tokens.data(), tokens.data() + tokens.size()) == expected);
        assert(mutable_tokens == tokens);

        // Match the salted block-by-block calls in waitRemotePrefixStatus.
        auto block_hash = salt;
        constexpr size_t stride = 2;
        for (size_t i = 0; i < tokens.size(); i += stride) {
            block_hash = rtp_llm::hashInt64Array(block_hash, tokens.data() + i, tokens.data() + i + stride);
        }
        assert(block_hash == expected);

        std::hash<int32_t> hasher;
        auto token_hash = salt;
        for (const auto token : tokens) {
            token_hash = rtp_llm::hashInt64Func(hasher, token_hash, token);
        }
        assert(token_hash == expected);
        assert(rtp_llm::hashInt64Array(salt, tokens.data(), tokens.data()) == salt);
    }
    std::cout << "HashUtil const-input tests passed\n";
}
