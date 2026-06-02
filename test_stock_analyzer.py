import unittest
from stock_analyzer import generate_stock_data, sma, ema, bollinger_bands, rsi, macd, compute_stats, STOCKS

class TestStockAnalyzer(unittest.TestCase):

    def setUp(self):
        self.data, self.name = generate_stock_data("AAPL", 180)
        self.closes = [d["close"] for d in self.data]

    def test_dataset_generated(self):
        self.assertGreater(len(self.data), 100)

    def test_ohlcv_structure(self):
        for col in ["date","open","high","low","close","volume"]:
            self.assertIn(col, self.data[0])

    def test_high_gte_low(self):
        for d in self.data:
            self.assertGreaterEqual(d["high"], d["low"])

    def test_sma_length(self):
        self.assertEqual(len(sma(self.closes, 20)), len(self.closes))

    def test_sma_first_window_none(self):
        result = sma(self.closes, 20)
        self.assertIsNone(result[0])
        self.assertIsNotNone(result[19])

    def test_ema_length(self):
        self.assertEqual(len(ema(self.closes, 20)), len(self.closes))

    def test_bb_length(self):
        u,m,l = bollinger_bands(self.closes)
        self.assertEqual(len(u), len(self.closes))

    def test_bb_upper_gte_lower(self):
        u,m,l = bollinger_bands(self.closes)
        for ui, li in zip(u, l):
            if ui and li:
                self.assertGreaterEqual(ui, li)

    def test_rsi_range(self):
        result = rsi(self.closes)
        for v in result:
            if v: self.assertGreater(v, 0); self.assertLess(v, 100)

    def test_macd_length(self):
        ml, ms, mh = macd(self.closes)
        self.assertEqual(len(ml), len(self.closes))

    def test_stats_keys(self):
        s = compute_stats(self.data)
        for k in ["total_return","sharpe","max_drawdown","win_rate","best_day","worst_day"]:
            self.assertIn(k, s)

    def test_win_rate_range(self):
        s = compute_stats(self.data)
        self.assertGreater(s["win_rate"], 0)
        self.assertLessEqual(s["win_rate"], 100)

    def test_unknown_ticker(self):
        with self.assertRaises(ValueError):
            generate_stock_data("XXXX")

    def test_all_tickers_work(self):
        for ticker in STOCKS:
            data, name = generate_stock_data(ticker, 90)
            self.assertGreater(len(data), 50)

if __name__ == "__main__":
    unittest.main()
