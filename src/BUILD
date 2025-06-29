load("//bazel:api.bzl", "mojo_library")

package(default_visibility = ["//visibility:public"])

mojo_library(
    name = "moe_kernel",
    srcs = ["moe_kernel.mojo"],
    deps = [
        "@mojo//:stdlib",
    ],
    visibility = ["//modular_hack:__subpackages__"],
)