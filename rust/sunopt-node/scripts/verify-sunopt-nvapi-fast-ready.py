#!/usr/bin/env python3
"""Offline structural test. Does not invoke any NVIDIA control."""
from pathlib import Path
import sys
import tempfile
from sunopt_nvapi_fast_ready import OLD_RAMP, NEW_RAMP, prepare_fast_nvapi_script

for name, original in [('legacy', OLD_RAMP),('fan-auto-enhanced', NEW_RAMP)]:
    # Test a valid original NVAPI source skeleton with the critical markers.
    source = '\n'.join(['# fanControl.readback=', '# curveControl.readback=', '# clockLock.readback=',
                        '# curveControl.restore.', '# fanControl.restore.', 'def sample():', original, '        pass'])
    with tempfile.TemporaryDirectory() as td:
        src = Path(td)/'original.py'; dest=Path(td)/'runtime.py'
        src.write_text(source,encoding='utf-8')
        prepare_fast_nvapi_script(src,dest)
        actual=dest.read_text(encoding='utf-8')
        assert 'last_smi = None' in actual and 'read_nvidia_smi(' not in actual
        assert src.read_text(encoding='utf-8') == source
        print(name+': PASS (original unchanged, fast readiness only in copy)')
with tempfile.TemporaryDirectory() as td:
    src=Path(td)/'bad.py'; dest=Path(td)/'runtime.py'
    src.write_text('print("unknown source")\n')
    try: prepare_fast_nvapi_script(src,dest)
    except RuntimeError: pass
    else: raise AssertionError('Unsupported version accepted')
    assert not dest.exists()
print('unsupported-source: FAIL CLOSED')
