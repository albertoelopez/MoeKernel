load("//bazel:api.bzl", "mojo_test")

package(default_visibility = ["//visibility:public"])

mojo_test(
    name = "benchmark_moe",
    srcs = ["benchmark_moe.mojo"],
    deps = [
        "//modular_hack/src:moe_kernel",
        "@mojo//:stdlib",
    ],
)

mojo_test(
    name = "official_moe_benchmark",
    srcs = ["official_moe_benchmark.mojo"],
    deps = [
        "//modular_hack/src:moe_kernel",
        "@mojo//:stdlib",
    ],
)

mojo_test(
    name = "comprehensive_performance",
    srcs = ["comprehensive_performance.mojo"],
    deps = [
        "//modular_hack/src:moe_kernel",
        "@mojo//:stdlib",
    ],
)