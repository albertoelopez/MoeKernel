load("//bazel:api.bzl", "mojo_test")

package(default_visibility = ["//visibility:public"])

mojo_test(
    name = "test_moe_kernel",
    srcs = ["test_moe_kernel.mojo"],
    deps = [
        "//modular_hack/src:moe_kernel",
        "@mojo//:stdlib",
    ],
)