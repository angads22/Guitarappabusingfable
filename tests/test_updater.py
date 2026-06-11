from fretmap import updater


def test_parse_version():
    assert updater.parse_version("v0.2.0") == (0, 2, 0)
    assert updater.parse_version("1.10.3") == (1, 10, 3)
    assert updater.parse_version("") == (0,)


def test_is_newer():
    assert updater.is_newer("v0.2.0", "0.1.0")
    assert updater.is_newer("1.0.0", "0.9.9")
    assert not updater.is_newer("0.2.0", "0.2.0")
    assert not updater.is_newer("v0.1.9", "0.2.0")


def test_asset_name_matches_release_naming():
    name = updater.asset_name(gui=True)
    assert name.startswith("fretmap-gui-")
    assert updater.platform_key() in name
    cli = updater.asset_name(gui=False)
    assert cli.startswith("fretmap-") and "gui" not in cli


def _release(tag, assets):
    return {
        "tag_name": tag,
        "html_url": f"https://github.com/x/y/releases/tag/{tag}",
        "assets": [
            {"name": n, "browser_download_url": f"https://dl/{n}"} for n in assets
        ],
    }


def test_find_update_returns_matching_asset():
    release = _release("v9.9.9", [updater.asset_name(gui=True), updater.asset_name(gui=False)])
    info = updater.find_update(release, gui=True, current="0.2.0")
    assert info is not None
    assert info.version == "9.9.9"
    assert info.asset_name == updater.asset_name(gui=True)
    assert info.download_url.endswith(info.asset_name)


def test_find_update_ignores_old_or_assetless_releases():
    assert updater.find_update(_release("v0.0.1", ["whatever"]), current="0.2.0") is None
    assert updater.find_update(_release("v9.9.9", ["wrong-name"]), current="0.2.0") is None
    assert updater.find_update(None) is None
