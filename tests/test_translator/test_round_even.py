"""roundEven, the hub's ties-to-even rounding, maps to each target's ties-to-even function."""

import pytest

import crosstl


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("directx", "round(a)"),
        ("slang", "round(a)"),
        ("wgsl", "round(a)"),
        ("metal", "rint(a)"),
        ("cuda", "rintf(a)"),
        ("hip", "rintf(a)"),
        ("opengl", "roundEven(a)"),
    ],
)
def test_round_even_maps_to_each_targets_ties_to_even_rounding(
    tmp_path, backend, expected
):
    path = tmp_path / "round.cgl"
    path.write_text(
        "shader S {\n    compute {\n        void main() {\n            float a = 2.5;\n"
        "            float b = roundEven(a);\n        }\n    }\n}\n"
    )
    out = crosstl.translate(str(path), backend=backend, format_output=False)
    assert expected in out
    if backend != "opengl":
        assert "roundEven" not in out


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_round_even_keeps_double_precision_and_vector_operands(tmp_path, backend):
    path = tmp_path / "round.cgl"
    path.write_text(
        "shader S {\n    compute {\n        void main() {\n            double a = 2.5;\n"
        "            double b = roundEven(a);\n            vec4 c = vec4(2.5);\n"
        "            vec4 d = roundEven(c);\n        }\n    }\n}\n"
    )
    out = crosstl.translate(str(path), backend=backend, format_output=False)
    assert "rint(a)" in out
    assert "rintf(a)" not in out
    assert "rintf(c)" not in out
    assert "rintf(value.x)" in out


HLSL_ROUND = """
RWStructuredBuffer<float> output : register(u0);
[numthreads(4, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    output[id.x] = round(output[id.x]);
}
"""


@pytest.mark.parametrize(
    "backend, expected",
    [
        ("cgl", "roundEven("),
        ("directx", "round("),
        ("slang", "round("),
        ("wgsl", "round("),
        ("metal", "rint("),
        ("cuda", "rintf("),
        ("hip", "rintf("),
        ("opengl", "roundEven("),
    ],
)
def test_hlsl_round_imports_as_ties_to_even_rounding(tmp_path, backend, expected):
    # HLSL round(2.5) is 2 (DXC lowers it to Round_ne); Metal round and CUDA roundf give 3.
    path = tmp_path / "round.hlsl"
    path.write_text(HLSL_ROUND)
    out = crosstl.translate(str(path), backend=backend, format_output=False)
    assert expected in out
    if backend in {"metal", "cuda", "hip"}:
        assert "round" not in out


def test_hlsl_user_function_named_round_is_not_renamed(tmp_path):
    path = tmp_path / "round.hlsl"
    path.write_text(
        "RWStructuredBuffer<float> output : register(u0);\n"
        "float round(float x, float step) { return floor(x / step) * step; }\n"
        "[numthreads(4, 1, 1)]\n"
        "void CSMain(uint3 id : SV_DispatchThreadID) {\n"
        "    output[id.x] = round(output[id.x], 0.5);\n"
        "}\n"
    )
    out = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert "round(output[id.x], 0.5)" in out
    assert "roundEven" not in out
