import importlib.util
import pathlib

_path = pathlib.Path(__file__).resolve().parent.parent / 'PopPUNK' / 'banner.py'
_spec = importlib.util.spec_from_file_location('banner', _path)
banner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(banner)


def test_banner_starts_and_ends_cleanly():
    assert banner.BANNER.splitlines()[0].strip().startswith('///o')
    assert banner.BANNER.endswith('\n')


def test_banner_includes_wordmark_and_credit():
    assert 'Made by John Lees' in banner.BANNER
    assert '|_|' in banner.BANNER
