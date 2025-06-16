# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['surface.py'],
    pathex=[],
    binaries=[],
    datas=[('test_image', 'test_image'), ('config.js', '.'), ('C:/Users/左欣雨/AppData/Local/Programs/Python/Python38/tcl/tcl8.6', 'tcl/tcl8.6'), ('C:/Users/左欣雨/AppData/Local/Programs/Python/Python38/tcl/tk8.6', 'tcl/tk8.6')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='surface',
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
    name='surface',
)
