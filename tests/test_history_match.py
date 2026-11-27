"""Game matching with older seasons (PBP mirror, no boxscores) and score-bug banners (2026-11-20)."""
import json

from agentdunker.film import director, season
from agentdunker.identity import games
from agentdunker.identity import pipeline as idp


def _sb(banner=None):
    return [{"teams": ["LAC", "DEN"], "scores": {"LAC": 0, "DEN": 2}, "period": 1, "clock": "11:39",
             "clock_s": 699.0, "banner": banner}]


def test_banner_names_the_kind_of_game():
    ctx = games.banner_context(_sb("SERIES TIED 2-2"))
    assert ctx["type"] == "playoffs" and ctx["game"] == 5 and "round" not in ctx
    ctx = games.banner_context(_sb("WEST FINALS - OKC LEADS 3-2"))
    assert (ctx["type"], ctx["round"], ctx["game"]) == ("playoffs", "3", 6)
    ctx = games.banner_context(_sb("EAST SEMIS DET LEADS 1-0"))
    assert (ctx["round"], ctx["game"]) == ("2", 2)
    assert games.banner_context(_sb("PLAY-IN TOURNAMENT"))["type"] == "playin"
    for other in ("ROCKETS: 12-2 RUN IN THE LAST 3:29", "THE DIFF: +22", "COAST 2 COAST TUESDAY", "3-POINTER",
                  "BUCKS AT THUNDER TOMORROW 6:30PM", None):
        assert games.banner_context(_sb(other)) is None, other


def test_playoff_ids_carry_round_and_game():
    ctx = {"type": "playoffs", "round": "1", "game": 5}
    assert games.fits_banner("0042400175", "playoffs", ctx)
    assert not games.fits_banner("0042400174", "playoffs", ctx)       # game 4
    assert not games.fits_banner("0042400275", "playoffs", ctx)       # second round
    assert not games.fits_banner("0022400515", "regular", ctx)
    assert games.fits_banner("0052400121", None, {"type": "playin"})  # type from the id prefix


def _patch(monkeypatch, cands, lineups, rosters, share):
    monkeypatch.setattr(games, "candidates", lambda teams, sb: [dict(c) for c in cands])
    monkeypatch.setattr(games, "rarity", lambda sb, banner=None: share)
    monkeypatch.setattr(games, "roster", lambda gid: rosters.get(gid))
    monkeypatch.setattr(idp, "_lineups", lambda gid, sb: lineups.get(gid))
    monkeypatch.setattr(games, "index", lambda: {c["game_id"]: {**c, "date": "2025-04-30"} for c in cands})


def _cand(gid, typ="regular"):
    return {"game_id": gid, "date": None, "type": typ, "home": "DEN", "away": "LAC", "n_fit": 1, "n_checked": 1}


NUMS = {"A": ["15", "27", "1"], "B": ["2", "8"]}


def test_jersey_confirmed_game_beats_games_without_a_boxscore(monkeypatch):
    lu = {"0022500695": {"DEN": {"15", "27", "1", "0", "5"}, "LAC": {"2", "8", "5", "4", "9"}}}
    _patch(monkeypatch, [_cand("0022500695"), _cand("0022400515")], lu, {"0022500695": {"DEN": {}}}, 0.004)
    gm = idp._match_game(_sb(), ["LAC", "DEN"], NUMS, idp.IdentityConfig())
    assert gm["status"] == "matched" and gm["game_id"] == "0022500695"


def test_unconfirmed_candidate_and_a_game_without_boxscore_stay_uncertain(monkeypatch):
    lu = {"0022500695": {"DEN": {"15", "27", "1", "0", "5"}, "LAC": {"2", "8", "5", "4", "9"}}}
    nums = {"A": ["15"], "B": ["2"]}                                    # two reads: not a lineup check
    _patch(monkeypatch, [_cand("0022500695"), _cand("0022400515")], lu, {"0022500695": {"DEN": {}}}, 0.004)
    gm = idp._match_game(_sb(), ["LAC", "DEN"], nums, idp.IdentityConfig())
    assert gm["status"] == "uncertain" and "1 without a boxscore" in gm["reason"]


def test_lone_game_without_boxscore_needs_a_rare_state(monkeypatch):
    sb = _sb("SERIES TIED 2-2")
    cands = [_cand("0042400175", "playoffs"), _cand("0022400515")]
    _patch(monkeypatch, cands, {}, {}, 0.0037)
    gm = idp._match_game(sb, ["LAC", "DEN"], NUMS, idp.IdentityConfig())
    assert gm["status"] == "matched" and gm["game_id"] == "0042400175"   # the banner removed the regular game
    assert 0.75 < gm["conf"] < 0.8 and gm["banner"]["game"] == 5
    _patch(monkeypatch, [_cand("0022400515")], {}, {}, 0.03)          # c13-like: not rare enough
    assert idp._match_game(_sb(), ["LAC", "DEN"], NUMS, idp.IdentityConfig())["status"] == "uncertain"


def _ident(**shooter):
    p = {"jersey": "8", "team": "B", "team_abbr": "LAC", "name": None, "person_id": None, "conf": 0.969}
    return {"players": {"10": {**p, **shooter}, "3": {"jersey": "27", "team": "A", "team_abbr": "DEN",
                                                       "name": None, "person_id": None, "conf": 0.9}}}


def _events(**official):
    off = {"person_id": 1627739, "player": "K. Dunn", "team": "LAC", "conf": 0.786,
           "match": {"name": False, "team": True, "made": True, "n_candidates": 1}}
    return {"shot": {"shooter_track": 10, "official": {**off, **official}}}


def test_shooter_named_from_the_play_by_play(monkeypatch):
    monkeypatch.setattr(season, "zh_name", lambda pid: {"name_zh": "克里斯·邓恩", "name_zh_short": "邓恩"})
    ident = _ident()
    out = director.official_shooter_fill(ident, _events())
    p = out["players"]["10"]
    assert (p["name"], p["person_id"], p["name_zh_short"], p["conf"]) == ("K. Dunn", 1627739, "邓恩", 0.786)
    assert ident["players"]["10"]["name"] is None                     # the input is not changed
    assert director.player_names(out)[10]["short"] == "邓恩"


def test_shooter_fill_only_for_a_pinned_shot_of_the_shooters_team(monkeypatch):
    monkeypatch.setattr(season, "zh_name", lambda pid: None)
    m = {"name": False, "team": True, "made": True, "n_candidates": 2}
    assert director.official_shooter_fill(_ident(), _events(match=m)) is None       # two field goals in the window
    assert director.official_shooter_fill(_ident(), _events(team="DEN")) is None    # other team's shot
    assert director.official_shooter_fill(_ident(), _events(conf=0.55)) is None   # below a matched game (0.6)
    assert director.official_shooter_fill(_ident(name="Kris Dunn", person_id=1627739), _events()) is None
    ident = _ident()
    ident["players"]["3"]["person_id"] = 1627739                       # someone else already is that player
    assert director.official_shooter_fill(ident, _events()) is None
    out = director.official_shooter_fill(_ident(), _events())
    assert out["players"]["10"]["name_zh"] is None and director.player_names(out)[10]["short"] == "Dunn"


def test_chinese_name_from_season_tables(tmp_path, monkeypatch):
    rows = [{"PLAYERID": 1627739, "FIRSTNAME": "克里斯", "LASTNAME": "邓恩"}, {"PLAYERID": 1629060, "FIRSTNAME": "八村垒",
                                                                          "LASTNAME": ""}]
    (tmp_path / "2025-26_RegularSeason_gravity.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False))
    monkeypatch.setattr(season, "DATA", tmp_path)
    season._zh_names.cache_clear()
    try:
        assert season.zh_name(1627739) == {"name_zh": "克里斯·邓恩", "name_zh_short": "邓恩"}
        assert season.zh_name("1629060") == {"name_zh": "八村垒", "name_zh_short": "八村垒"}
        assert season.zh_name(1) is None
    finally:
        season._zh_names.cache_clear()
