load("//bazel:api.bzl", "mojo_library")

package(default_visibility = ["//visibility:public"])

# Optimized MOE kernel for MAX integration
mojo_library(
    name = "moe_max_kernel",
    srcs = ["moe_max_kernel.mojo"],
    deps = [
        "@mojo//:compiler",
        "@mojo//:layout",
        "@mojo//:stdlib",
        "//modular_hack/src:moe_kernel",  # Our base MOE implementation
    ],
    visibility = ["//modular_hack:__subpackages__"],
)

# Python tests and examples
py_binary(
    name = "moe_max_model",
    srcs = ["moe_max_model.py"],
    deps = [
        # MAX dependencies would go here in a real setup
        # "@max//python:max",
    ],
    data = [":moe_max_kernel"],  # Include the Mojo kernel
)