# Input schema

`prices/PAIR.csv`: `Date, Open, High, Low, Close`. Dates must be unique and prices numeric; malformed date rows are removed explicitly. Currency-pair names are never substituted (USDCNY is not USDCNH).

`features/PAIR.csv`: first column `Date`; remaining columns numeric selected predictors. Used with `--selection supplied`.

`macro/PAIR/COUNTRY/*.csv`: first column a date, second column a numeric macro series. Used with `--selection rebuilt`. The historical Appendix A reads only the first two columns; a multivariate macro workbook must be split deliberately if all columns are needed.

Included `data/demo/` files are generated independently by `scripts/generate_demo.py` (seed 2025). The prices are synthetic and are not market history. Data-provider source workbooks are not distributed, since their redistribution rights have not been established in the supplied materials.
