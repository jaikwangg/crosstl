"""Written ``in`` array parameters keep value semantics on CUDA and HIP (#2251).

HLSL passes an array parameter without ``out`` or ``inout`` by value, and the HLSL importer records it as ``in``. A
C++ array parameter decays to a pointer, so a write in the callee used to reach the caller's array. CUDA and HIP now
receive a written ``in`` array under another name and copy it into a local that keeps the source name, as Metal does.
Unwritten ``in`` arrays, ``out`` and ``inout`` arrays and arrays without a qualifier keep their declaration.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.cuda_codegen import CudaCodeGen
from crosstl.translator.codegen.hip_codegen import HipCodeGen

GENERATORS = {"cuda": CudaCodeGen, "hip": HipCodeGen}

ISSUE_HLSL = """
RWStructuredBuffer<float> results : register(u0);
float first(float v[2]) { v[0] = 3.0; return v[0] + v[1]; }
[numthreads(1,1,1)] void main() { float a[2] = { 1.0, 2.0 }; results[1] = first(a); results[2] = a[0]; }
"""

PROBE = """
shader main {
    buffer float* results;

    float written(in float v[2]) {
        v[0] = 3.0;
        return v[0] + v[1];
    }

    float grid(in float m[2][3]) {
        m[1][2] += 4.0;
        return m[1][2] + m[0][0];
    }

    float bumped(in int c[2]) {
        c[1]++;
        return float(c[1]);
    }

    float unwritten(in float u[2]) {
        return u[0] + u[1];
    }

    void fill(inout float w[2]) {
        w[1] = 7.0;
    }

    float through_inout(in float n[2]) {
        fill(n);
        return n[1];
    }

    float through_in(in float k[2]) {
        return written(k);
    }

    void reset(out float o[2]) {
        o[0] = 0.0;
        o[1] = 0.0;
    }

    float unqualified(float p[2]) {
        p[0] = 5.0;
        return p[0];
    }

    compute {
    @ numthreads(1, 1, 1)
    void main() {
        float a[2] = {1.0, 2.0};
        float g[2][3] = {{1.0, 0.0, 0.0}, {0.0, 0.0, 2.0}};
        int c[2] = {1, 1};
        results[0] = written(a);
        results[1] = grid(g);
        results[2] = bumped(c);
        results[3] = through_inout(a);
        results[4] = through_in(a);
        results[5] = unwritten(a);
        results[6] = a[0] + a[1] + g[1][2] + float(c[1]);
    }
    }
}
"""


def _generate(source, backend):
    return GENERATORS[backend]().generate(crosstl.translator.parse(source))


def _signature(code, name):
    match = re.search(rf"__device__ \w+ {name}\(([^)]*)\)", code)
    assert match, name
    return match.group(1)


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_hlsl_array_written_by_the_callee_is_copied(tmp_path, backend):
    path = tmp_path / "probe.hlsl"
    path.write_text(ISSUE_HLSL, encoding="utf-8")
    code = crosstl.translate(str(path), backend=backend)
    assert _signature(code, "first") == "float v_crosstl_input[2]"
    assert "float v[2];" in code
    assert re.search(
        r"for \(int crosstl_v_i0 = 0; crosstl_v_i0 < int\(2\); \+\+crosstl_v_i0\) "
        r"\{ v\[crosstl_v_i0\] = v_crosstl_input\[crosstl_v_i0\]; \}",
        code,
    )


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_written_in_arrays_are_received_under_another_name(backend):
    code = _generate(PROBE, backend)
    assert _signature(code, "written") == "float v_crosstl_input[2]"
    assert _signature(code, "grid") == "float m_crosstl_input[2][3]"
    assert _signature(code, "bumped") == "int c_crosstl_input[2]"
    # Passing the array to an inout parameter writes it.
    assert _signature(code, "through_inout") == "float n_crosstl_input[2]"
    assert "float m[2][3];" in code
    assert (
        "m[crosstl_m_i0][crosstl_m_i1] = m_crosstl_input[crosstl_m_i0][crosstl_m_i1];"
        in code
    )


@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_other_array_parameters_keep_their_declaration(backend):
    code = _generate(PROBE, backend)
    assert _signature(code, "unwritten") == "float u[2]"
    # The callee copies its own in array, so passing one along is not a write.
    assert _signature(code, "through_in") == "float k[2]"
    assert "w[2]" in _signature(code, "fill")
    assert "o[2]" in _signature(code, "reset")
    assert _signature(code, "unqualified") == "float p[2]"
    assert "crosstl_input" not in "".join(
        _signature(code, name)
        for name in ("unwritten", "through_in", "fill", "reset", "unqualified")
    )


def _device_functions(code):
    """The source of every ``__device__`` helper, which is also host C++."""
    start = code.index("__device__ float written(")
    end = code.index('extern "C"') if 'extern "C"' in code else code.index("__global__")
    return code[start:end]


HOST_MAIN = """
#include <cstdio>
int main() {
    float a[2] = {1.0f, 2.0f};
    float g[2][3] = {{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 2.0f}};
    int c[2] = {1, 1};
    float r[7];
    r[0] = written(a);
    r[1] = grid(g);
    r[2] = bumped(c);
    r[3] = through_inout(a);
    r[4] = through_in(a);
    r[5] = unwritten(a);
    r[6] = a[0] + a[1] + g[1][2] + float(c[1]);
    for (float value : r) std::printf("%g ", value);
    return 0;
}
"""


# The helpers compile as host C++ once __device__ is defined away, so the copies can be checked without a GPU.
@pytest.mark.parametrize("backend", sorted(GENERATORS))
def test_callee_writes_do_not_reach_the_caller(tmp_path, backend):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("no C++ compiler is installed")
    code = _generate(PROBE, backend)
    program = tmp_path / "probe.cpp"
    program.write_text(
        "#define __device__\n" + _device_functions(code) + HOST_MAIN, encoding="utf-8"
    )
    binary = tmp_path / "probe"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-w", str(program), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
    # Direct3D: no call changes a, g or c, so the last sum is 1 + 2 + 2 + 1.
    assert ran.stdout.split() == ["5", "7", "2", "7", "5", "3", "6"]
