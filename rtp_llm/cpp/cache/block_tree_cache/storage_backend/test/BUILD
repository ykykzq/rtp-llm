load("//:def.bzl", "cc_test_wrapper", "copts")
load("//bazel:arch_select.bzl", "torch_deps")

cc_test(
    name = "remote_write_ledger_test",
    srcs = ["RemoteWriteLedgerTest.cc"],
    copts = ["-std=c++17"],
    linkopts = ["-pthread"],
    deps = ["//rtp_llm/cpp/cache/block_tree_cache:remote_write_ledger"],
)

cc_test_wrapper(
    name = "storage_backend_test",
    exec_properties = {"gpu": "L20_CU13"},
    testonly = True,
    srcs = ["StorageBackendTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache/block_tree_cache/test:bounded_thread_test_utils",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ] + torch_deps(),
    tags = ["L20_CU13"],
)
