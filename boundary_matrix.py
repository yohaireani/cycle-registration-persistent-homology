
def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def concatenatestring(x, y):
    return str(x) + "," + str(y)


def sortbyfilt(c):
    return c.filtration


def sortbydeg(c):
    return c.degree


class Col:
    # boundary matrix column
    def __init__(self, simplex, filtration=.0):
        '''
        :param simplex: list of integers
        :param filtration: double
        '''
        self.simplex = simplex[:]
        self.filtration = filtration
        self.degree = len(simplex) - 1
        self.low = -1
        if len(simplex) > 1:
            self.low = max(simplex)

    def __iadd__(self, c):
        self.simplex = list(set(self.simplex).symmetric_difference(set(c.simplex)))
        if not self.simplex:
            self.low = -1
        else:
            print(self.simplex)
            self.low = max(self.simplex)
            print(['bla', self.low])

    def add(self, c):
        self.simplex = list(set(self.simplex).symmetric_difference(set(c.simplex)))
        if not self.simplex:
            self.low = -1
        else:
            self.low = max(self.simplex)


class BoundaryMatrix:
    # boundary matrix
    def __init__(self, rowfiltvals=None):
        self.numcols = 0
        self.cols = []
        if rowfiltvals:
            self.rowfiltvals = rowfiltvals[:]
        else:
            self.rowfiltvals = []
        self.reduced = False
        self.cycles = {}
        # negative simplex to positive simplex dictionary
        self.neg2pos_dict = {}
        # positive simplex to generator index in the generators list
        self.pos2idx = {}
        self.low_dict = {}
        self.intervals = []
        self.gen_indices = []
        self.generators = []
        self.pairs = []

    def add_col(self, simplex, filtration, cycles=None):
        c = Col(simplex, filtration)
        c.simplex.sort()
        self.cols.append(c)
        if not cycles:
            self.cycles[self.numcols] = {self.numcols}
        else:
            self.cycles[self.numcols] = set(cycles)
        if c.low in self.low_dict:
            self.low_dict[c.low].append(self.numcols)
        else:
            self.low_dict[c.low] = [self.numcols]
        self.numcols += 1
        self.reduced = False

    def reduce(self):
        if self.numcols < 2:
            self.reduced = True
            return
        cols = self.cols
        col_idx = 0
        cnt = 0
        while col_idx < self.numcols:
            lowi = cols[col_idx].low
            if lowi == -1:  # all-zeros column - potential generator of a (k+1)-cycle
                col_idx += 1
                continue
            tmp = self.low_dict[lowi][:]
            idx = diff(tmp, [col_idx])
            # min(idx) is the index of the leftest column
            if len(idx) == 0 or min(idx) > col_idx:
                # new part - interval of a k-cycle
                if self.rowfiltvals[lowi] < cols[col_idx].filtration:
                    self.intervals.append((self.rowfiltvals[lowi], cols[col_idx].filtration))
                    self.neg2pos_dict[col_idx] = lowi
                    self.pos2idx[lowi] = cnt
                    # save generator
                    self.gen_indices.append(col_idx)
                    cnt += 1
                col_idx += 1
                continue
            # dictionary update
            self.low_dict[lowi] = idx
            # column addition
            m = min(idx)
            cols[col_idx].add(cols[m])
            self.cycles[col_idx] = self.cycles[col_idx].union(self.cycles[m])
            new_low = cols[col_idx].low

            if new_low in self.low_dict:
                self.low_dict[new_low].append(col_idx)
            else:
                self.low_dict[new_low] = [col_idx]
                # new part - interval of a k-cycle
                if self.rowfiltvals[new_low] < cols[col_idx].filtration:
                    self.intervals.append((self.rowfiltvals[new_low], cols[col_idx].filtration))
                    self.neg2pos_dict[col_idx] = new_low
                    self.pos2idx[new_low] = cnt
                    # save generator index
                    self.gen_indices.append(col_idx)
                    cnt += 1
                #
                col_idx += 1
        self.reduced = True

    def make_canonical(self):
        cols = self.cols
        col_idx = 0
        while col_idx < self.numcols:
            if (cols[col_idx].low == -1) or (len(cols[col_idx].simplex) == 1):
                col_idx += 1
                continue
            simplex = cols[col_idx].simplex
            simplex.sort(reverse=True)
            cnt = 1
            while 1:
                if cnt == len(simplex):
                    break
                lead1 = simplex[cnt]
                if lead1 not in self.low_dict:
                    cnt += 1
                    continue
                idx = self.low_dict[lead1][:]
                if min(idx) > col_idx:
                    cnt += 1
                    continue
                m = min(idx)
                cols[col_idx].add(cols[m])
                simplex = cols[col_idx].simplex
                simplex.sort(reverse=True)
            col_idx += 1

    def reduce_canonical(self):
        self.reduce()
        self.make_canonical()

    def get_intervals(self):
        if not self.reduced:
            self.reduce_canonical()
            for x in self.gen_indices:
                self.generators.append(self.cols[x].simplex[:])
        return self.intervals, self.generators, self.neg2pos_dict, self.pos2idx

    def reducecol(self, col):
        maxchain = 0
        genlist = []
        while 1:
            if col.low == -1:
                break
            if col.low not in self.low_dict:
                break
            ind = self.low_dict[col.low][0] # if self is reduced, every key has one value
            col.add(self.cols[ind])
            genlist.append(ind)
            if ind > maxchain:
                maxchain = ind
        return col, maxchain, genlist



