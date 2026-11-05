"""A function named ``read``, ``write`` or ``read_write`` is a function.

The Metal lexer gives ``read``, ``write`` and ``read_write`` their own tokens
for ``access::read`` and the other texture access qualifiers. The parser took
the name in ``uint read(Wrapper value) { ... }`` as a trailing type qualifier,
so the definition was parsed as a declaration and failed with
``Expected SEMICOLON, got LBRACE`` (#2144). An access name directly followed
by ``(`` is now read as a function name, in a definition, a forward
declaration and a call, while the texture access qualifiers keep working.
"""

import shutil
import subprocess
import textwrap

import pytest

import crosstl

ISSUE_KERNEL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;
    struct Wrapper { uchar payload; };
    namespace wrappers {
    uint read(Wrapper value) { return uint(value.payload); }
    }
    kernel void test(const device uchar* values [[buffer(0)]],
                     device uint* output [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
        using Value = Wrapper;
        output[i] = wrappers::read(((const device Wrapper*)values)[i]);
    }
    """)

KERNEL = textwrap.dedent("""
    #include <metal_stdlib>
    using namespace metal;

    uint read(uint value);
    uint read(uint value) { return value * 3u; }
    uint read(uint a, uint b) { return a * b + 1u; }
    uint write(uint value) { return value + 7u; }
    uint read_write(uint value) { return value ^ 5u; }

    namespace helpers {
    uint write(uint a, uint b) { return a - b; }
    }

    kernel void k(device uint* out [[buffer(0)]],
                  texture2d<float, access::read> src [[texture(0)]],
                  texture2d<float, access::write> dst [[texture(1)]],
                  texture2d<float, access::read_write> rw [[texture(2)]],
                  uint2 gid [[thread_position_in_grid]])
    {
        uint i = gid.x;
        out[i] = read(i) + read(i, 2u) + write(i) + read_write(i)
            + helpers::write(i, 1u);
        dst.write(src.read(gid) + rw.read(gid), gid);
    }
    """)


def _translate(tmp_path, source, backend, name="access_names.metal"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("backend", ["metal", "opengl", "directx"])
def test_namespaced_read_function_translates(tmp_path, backend):
    code = _translate(tmp_path, ISSUE_KERNEL, backend)
    assert "uint read(Wrapper value)" in code
    assert "read(" in code.split("uint read(Wrapper value)", 1)[1]


def test_access_named_functions_are_imported(tmp_path):
    code = _translate(tmp_path, KERNEL, "cgl")
    for signature in (
        "uint read(uint value)",
        "uint read(uint a, uint b)",
        "uint write(uint value)",
        "uint read_write(uint value)",
        "uint write(uint a, uint b)",
    ):
        assert signature in code
    assert "read(i) + read(i, 2u) + write(i) + read_write(i) + write(i, 1u)" in code


def test_texture_access_qualifiers_are_kept(tmp_path):
    code = _translate(tmp_path, KERNEL, "cgl")
    assert "image2D src @texture(0) @readonly" in code
    assert "image2D dst @texture(1) @writeonly" in code
    assert "image2D rw @texture(2) @readwrite" in code
    assert "imageStore(dst, gid, imageLoad(src, gid) + imageLoad(rw, gid));" in code


_VALIDATORS = {
    "opengl": (
        "access_names.comp",
        [["glslangValidator", "-S", "comp", "access_names.comp"]],
    ),
    "directx": (
        "access_names.hlsl",
        [["dxc", "-T", "cs_6_2", "-E", "CSMain", "access_names.hlsl"]],
    ),
}


@pytest.mark.parametrize("backend", sorted(_VALIDATORS))
def test_access_named_functions_reach_every_target(tmp_path, backend):
    file_name, commands = _VALIDATORS[backend]
    code = _translate(tmp_path, KERNEL, backend)
    missing = [command[0] for command in commands if shutil.which(command[0]) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not installed")
    (tmp_path / file_name).write_text(code, encoding="utf-8")
    for command in commands:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300, cwd=tmp_path
        )
        assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
