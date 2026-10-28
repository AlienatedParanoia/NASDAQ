# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# the index membership CSVs are read at runtime; without this they are simply
# absent from the exe and the universe fails to load (see paths.resource_dir)
datas = [
    ('data/constituents/*.csv', 'data/constituents'),
    ('bots.yaml', '.'),          # bot definitions are read at runtime
]
binaries = []
hiddenimports = ['_cffi_backend', 'peewee', 'multitasking', 'yaml']
tmp_ret = collect_all('yfinance')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('curl_cffi')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('certifi')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# exchange_calendars ships CSV holiday/precomputed data; without collect_all the
# frozen build imports fine and then fails at the first calendar lookup
tmp_ret = collect_all('exchange_calendars')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6', 'PyQt5', 'matplotlib'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='nasdaq-advisor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='nasdaq-advisor',
)
