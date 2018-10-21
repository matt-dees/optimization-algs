class CSVFormatter:
    def __init__(self):
        self._table = []

    def add_row(self, row):
        self._table.append(row)

    def clear_table(self):
        self._table = []

    def export_table(self, name):
        with open(name, "w") as f:
            for row in self._table:
                f.write(", ".join([str(ele) for ele in row]))
                f.write("\n")