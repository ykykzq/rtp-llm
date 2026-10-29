cc_test(
    name = "resolver_test",
    srcs = ["resolver_test.cc"],
    copts = ["-O2", "-U_FORTIFY_SOURCE", "-D_FORTIFY_SOURCE=2", "-pthread"],
    linkopts = ["-pthread"],
    deps = ["@curl//:curl"],
)

cc_binary(
    name = "curl_symbol_probe.so",
    testonly = True,
    srcs = ["symbol_probe.cc"],
    linkshared = True,
    deps = ["@curl//:curl"],
)

py_test(
    name = "symbol_isolation_test",
    srcs = ["symbol_isolation_test.py"],
    args = [
        "$(location :curl_symbol_probe.so)",
        "$(location :conflicting_curl.so)",
    ],
    data = [":curl_symbol_probe.so", ":conflicting_curl.so"],
)

cc_binary(
    name = "conflicting_curl.so",
    testonly = True,
    srcs = ["conflicting_curl.cc"],
    linkshared = True,
)

cc_import(
    name = "conflicting_sdk",
    testonly = True,
    shared_library = ":conflicting_curl.so",
)

cc_binary(
    name = "curl_sdk_linked_probe.so",
    testonly = True,
    srcs = ["sdk_linked_probe.cc"],
    linkshared = True,
    deps = [":conflicting_sdk", "@curl//:curl"],
)

py_test(
    name = "sdk_linked_symbol_isolation_test",
    srcs = ["symbol_isolation_test.py"],
    main = "symbol_isolation_test.py",
    args = [
        "$(location :curl_sdk_linked_probe.so)",
        "$(location :conflicting_curl.so)",
    ],
    data = [":curl_sdk_linked_probe.so", ":conflicting_curl.so"],
)
