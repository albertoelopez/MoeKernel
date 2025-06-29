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