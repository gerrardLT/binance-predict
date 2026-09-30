from binance_predict.services.process_recovery import extract_process_recovery


def curves(confirm=120_000, q=.25):
    return ([{'t': 0, 'v': .1}, {'t': 30_000, 'v': .20}, {'t': confirm, 'v': q}],
            [{'t': 0, 'v': 100.}, {'t': 30_000, 'v': 101.}, {'t': confirm, 'v': 100.5}])


def test_process_confirmation_boundaries_and_prefix():
    down, btc = curves()
    got = extract_process_recovery(0, 100., down, btc)
    assert got and got['q'] == .25 and got['touch_ts'] == 30_000
    assert got['recovery_frac'] > .35
    assert extract_process_recovery(0, 100., down, btc, max_ts=90_000) is None
    for t, q, passes in [(179_999, .35, True), (180_000, .25, False), (120_000, .350001, False)]:
        d, b = curves(t, q)
        assert (extract_process_recovery(0, 100., d, b) is not None) is passes
    # 低价触达仍保留；过滤的是首次双确认价，0.15贴线拒绝。
    for q, passes in [(.149999, False), (.15, False), (.150001, True)]:
        d, b = curves(q=q)
        d[1]['v'] = .12
        assert (extract_process_recovery(0, 100., d, b) is not None) is passes
    # 第一确认超价，随后回落不得追入。
    d, b = curves(q=.36)
    d.append({'t': 135_000, 'v': .30})
    b.append({'t': 135_000, 'v': 100.4})
    assert extract_process_recovery(0, 100., d, b) is None
