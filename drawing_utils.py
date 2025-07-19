from typing import Callable, Optional

import numpy as np
from tabulate import tabulate
from enum import Enum


class TableStyle(Enum):
    Default = 1
    Booktabs = 2


def drawing_tabel_default(tabulars: list[list[list]], caption, label="test", tabular_heads=None):
    tabulars_code = list()
    if tabular_heads is not None:
        assert len(tabular_heads) == len(tabulars)

    for p, tab in enumerate(tabulars):
        col_num = len(tab[0])
        align = "c" * col_num
        rows = []
        if tabular_heads:
            rows.append(r"\multicolumn{%d}{c}{%s}" % (col_num, tabular_heads[p]))
        for row in tab:
            row = map(str, row)  # lambda x: r"\text{%s}" % x
            rows.append(" & ".join(row))
        content = "\\\\ \\hline \n".join(rows)
        tabular = r"""
        \begin{tabular}{%s}
            %s
          \end{tabular}
        """ % (align, content)
        tabulars_code.append(tabular)

    tabulars_code = "\n".join(tabulars_code)

    tabel = r"""
    \begin{table*}
      \centering
      %s
      \caption{%s}
      \label{tab:%s}
    \end{table*}
        """ % (tabulars_code, caption, label)
    print(tabel)


def is_bit_set(number, position):
    """
    判断数字的第 position 位是否为1。
    position 从0开始，即最低位为第0位。
    """
    if (number & (1 << position)) != 0:
        return 1
    else:
        return 0

def drawing_tabel_booktabs(tabulars: list[list[list]], caption, label="test", tabular_heads=None, control_code: int=7):
    tabulars_code = list()
    if tabular_heads is not None:
        assert len(tabular_heads) == len(tabulars)

    for p, tab in enumerate(tabulars):
        col_num = len(tab[0])
        align = "c" * col_num
        rows = []

        for row in tab:
            row = map(str, row)  # lambda x: r"\text{%s}" % x
            rows.append(" & ".join(row))
        neckline = [("", r"\toprule"), (r"\\", r"\\ \midrule"), (r"\\", r"\\ \bottomrule")]
        if len(rows) >= 1:
            rows[0] = neckline[0][is_bit_set(control_code, 2)] + "\n" + rows[0]
        if len(rows) >= 2:
            rows[1] = neckline[1][is_bit_set(control_code, 1)] + "\n" + rows[1]
        if len(rows) > 4:
            for i in range(1, len(rows) - 1):
                rows[i] = rows[i] + r"\\"
        if len(rows) > 3:
            rows[-1] = rows[-1] + neckline[2][is_bit_set(control_code, 0)]

        if tabular_heads:
            rows.append(r"\multicolumn{%d}{c}{%s}" % (col_num, tabular_heads[p]))
        content = "\n".join(rows)
        tabular = r"""
        \begin{tabular}{%s}
            %s
          \end{tabular}
        """ % (align, content)
        tabulars_code.append(tabular)

    tabulars_code = "\n".join(tabulars_code)

    tabel = r"""
    \begin{table*}
      \centering
      %s
      \caption{%s}
      \label{tab:%s}
    \end{table*}
        """ % (tabulars_code, caption, label)
    print(tabel)


def drawing_tabel(tabulars: list[list[list]], caption, label="test", tabular_heads=None,
                  style: TableStyle = TableStyle.Default, control_code=7):
    if style == TableStyle.Default:
        return drawing_tabel_default(tabulars, caption, label, tabular_heads)
    elif style == TableStyle.Booktabs:
        return drawing_tabel_booktabs(tabulars, caption, label, tabular_heads, control_code=control_code)


def transpose(tabular: list[list]) -> list[list]:
    ts = np.array(tabular).transpose()
    return ts.tolist()


def add_hat(tabular: list[list], row: list, col: list):
    assert len(row) == len(tabular[0]) + 1, str(row) + str(tabular[0])
    assert len(tabular) == len(col), str(tabular) + str(col)
    tab = list([row])
    for p, c in enumerate(tabular):
        tab.append([col[p]] + c)
    return tab


def remark_max(tabular: list[list], func, func2=lambda x: x):
    """
    按行对最大值进行操作
    :param func2:
    :param tabular:
    :param func:
    :return:
    """
    rows = list()
    for row in tabular:
        max_element = max(row)
        row = [func(max_element) if i == max_element else func2(i)
               for i in row]
        rows.append(row)
    return rows


def merge_tabulars(tabular1: list[list], tabular2: list[list], mergeing_func: Callable[[str, str], str]):
    tabs = list()
    x, y = len(tabular1), len(tabular1[0])
    for i in range(x):
        tabs.append(list())
        for j in range(y):
            tabs[-1].append(mergeing_func(tabular1[i][j], tabular2[i][j]))
    return tabs


def variable2title(x: str):
    """
    把形如 a_b_c 的变量名转变为 A B C形式的字符串。
    :param x:
    :return:
    """
    return " ".join([i.capitalize() for i in x.split("_")])


def get_abbr_from_title(x: str):
    return "".join([i[0] for i in x.split(" ")])


class Tabular:
    def __init__(self, tabular: Optional[list[list]] = None):
        self.tabular = tabular

    def from_dicts(self, dicts: list[dict], labels, head, var2title=True, show_abbr=False, keys=None):
        if keys is None:
            keys = dicts[0].keys()
        if var2title:
            if show_abbr:
                self.tabular = [[head] + [get_abbr_from_title(variable2title(i)) for i in keys]]
            else:
                self.tabular = [[head] + [variable2title(i) for i in keys]]
        else:
            self.tabular = [[head] + list(keys)]
        for p, d in enumerate(dicts):
            self.tabular.append([labels[p]] + [d[key] for key in keys])
        return self

    def size(self):
        return len(self.tabular), len(self.tabular[0])

    def transpose(self):
        self.tabular = transpose(self.tabular)

    def add_hat(self, row: list, col: list):
        self.tabular = add_hat(self.tabular, row, col)

    def merge_tabulars(self, tabular2, mergeing_func: Callable[[str, str], str]):
        self.tabular = merge_tabulars(self.tabular, tabular2.tabular, mergeing_func)

    def remark_max(self, func, func2=lambda x: x):
        self.tabular = remark_max(self.tabular, func, func2)

    def print(self):
        print(tabulate(self.tabular[1:], headers=self.tabular[0], tablefmt="grid"))

    def print_tabel(self, caption, label, style: TableStyle = TableStyle.Default, control_code=7):
        if isinstance(self.tabular[0][0], str):
            print(drawing_tabel([self.tabular], caption, label, style=style, control_code=control_code))
        else:
            print("#")
            print(drawing_tabel(self.tabular, caption, label, style=style, control_code=control_code))


if __name__ == "__main__":
    Tabular().from_dicts([
        {'note_compression_ratio': 0.7319381312848035, 'token_using_in_learning': 1221083,
         'inference_cost': 266935.47777777776},
        {'note_compression_ratio': 0.7319381312848035, 'token_using_in_learning': 1221083,
         'inference_cost': 266935.47777777776}
    ], labels=["rewrite", "w/o noting"], head="metric").print()
