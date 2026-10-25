import unittest
import numpy as np
import pandas as pd
from fxrisk.features import make_target
from fxrisk.data import normalize_prices
from fxrisk.model import risk_classes


class ResearchRules(unittest.TestCase):
    def test_target_window_contains_exact_documented_returns(self):
        close = np.exp(np.cumsum(np.linspace(-.02, .03, 80)**2))
        frame = pd.DataFrame({'Date': pd.bdate_range('2020-01-01', periods=80), 'Close': close})
        result = make_target(frame)
        returns = pd.Series(close).pct_change()
        # At t=30, 20 observations ending at t+5 contain returns 16..35.
        self.assertAlmostEqual(result.loc[frame.Date[30]], returns.iloc[16:36].std())

    def test_currency_price_rows_keep_correct_date_column(self):
        frame = pd.DataFrame({'代码':['USDCNH.FX']*2, '日期':['2024-01-02','invalid'],
                              '开盘价(元)':[7,7], '最高价(元)':[8,8], '最低价(元)':[6,6], '收盘价(元)':[7,7]})
        result = normalize_prices(frame)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.Date.iloc[0], pd.Timestamp('2024-01-02'))

    def test_jpycny_thresholds_are_fixed_not_test_quantiles(self):
        np.testing.assert_array_equal(risk_classes(np.array([.001,.005,.01]), 'JPYCNY'), [0,1,2])


if __name__ == '__main__':
    unittest.main()
