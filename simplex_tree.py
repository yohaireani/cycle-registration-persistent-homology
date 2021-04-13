from boundary_matrix import BoundaryMatrix

def getkey(node):
    return node.val

def getfiltration(node):
    return node.filtration

def list2str(list):
    str_list = [str(x) for x in list]
    return '-'.join(str_list)

def str2list(str):
    x = str.split('-')
    list = [int(a) for a in x]
    return list

class TreeNode:
    def __init__(self, val, filtration, parent):
        self.val = val
        self.filtration = filtration
        self.parent = parent
        self.children = {}
        self.index = 0

    def add_child(self, node):
        if node.val not in self.children:
            self.children[node.val] = node

    def __getitem__(self, key):
        node = None
        for x in self.children:
            if self.children[x].val == key:
                node = self.children[x]
                break
        return node

    # def __iter__(self):
    #     return self.children
    #
    # def __next__(self):
    #     if self.index == len(self.children) - 1:
    #         raise StopIteration
    #     return self.children[self.index+1]


class SimplexTree:
    def __init__(self, simplexes=None):
        self.root = TreeNode(-1, float('-inf'), None)
        # self.boundary_matrix = BoundaryMatrix()
        self.simplex_dict = {}
        self.lists = []
        # self.pairs = []
        self.dimension = 0
        if simplexes:
            for simplex in simplexes:
                self.add_simplex(simplex)

    def __add_simplex__(self, simplex, filtration=0.0, parent=None, dim=0):
        if not parent:
            parent = self.root
        cnt = 1
        for i in simplex:
            if i not in parent.children:
                new_node = TreeNode(i, filtration, parent)
                parent.add_child(new_node)
                if dim not in self.simplex_dict:
                    self.simplex_dict[dim] = [new_node]
                else:
                    self.simplex_dict[dim].append(new_node)

                if dim >= 0:
                    if dim > self.dimension:
                        self.dimension = dim
                    if len(self.lists) >= dim + 1:
                        if i in self.lists[dim]:
                            self.lists[dim][i].append(new_node)
                        else:
                            self.lists[dim][i] = [new_node]
                    else:
                        self.lists.append({})
                        self.lists[dim][i] = [new_node]
            else:
                if parent.children[i].filtration > filtration:
                    parent.children[i].filtration = filtration

            sub_simplex = simplex[cnt:]
            if sub_simplex:
                self.__add_simplex__(sub_simplex, filtration, parent[i], dim+1)
            cnt += 1

    def add_simplex(self, simplex, filtration=0.0):
        '''
        simplex: list of vertices
        filtration: filtration value
        '''
        simplex.sort()
        self.__add_simplex__(simplex, filtration)

        # update filtration values of co_faces
        cofaces = self.get_cofaces(simplex)
        for x in cofaces:
            x.filtration = filtration

    def node2simplex(self, node):
        simplex = []
        while node.val != -1:
            simplex.append(node.val)
            node = node.parent
        simplex.reverse()
        return simplex

    def find_simplex(self, simplex):
        simplex.sort()
        node = self.root
        for i in simplex:
            node = node.children[i]
        return node

    def get_facets(self, simplex):
        simplex.sort()
        facets = []
        node = self.root
        cnt = 0
        for i in simplex:
            sub_simplex = simplex[:]
            sub_simplex.pop(cnt)
            facets.append(self.find_simplex(sub_simplex))
            cnt += 1
        return facets

    def get_cofaces(self, simplex):
        simplex.sort()
        k = len(simplex) - 1
        last_vertex = simplex[-1]

        cofaces = []
        # iterating over lists with dimension at least k
        for d in range(k, len(self.lists)):
            tmp_list = self.lists[d]
            if last_vertex in tmp_list:
                lv_dic = tmp_list[last_vertex] # last vertex dictionary
                for node in lv_dic:
                    if node.parent:
                        node_it = node.parent
                        flag = 0
                        cnt = len(simplex) - 2
                        while cnt > -1 and flag == 0:
                            while node_it.val != simplex[cnt]:
                                node_it = node_it.parent
                                if not node_it:
                                    flag = 1
                                    break
                            cnt -= 1
                        if flag == 0:
                            q = [node]
                            while q:
                                node = q.pop(0)
                                cofaces.append(node)
                                q = q + [node.children[x] for x in node.children]

        return cofaces

    def get_simplexes(self, dim):
        simplex_list = []
        if dim in self.simplex_dict:
            for node in self.simplex_dict[dim]:
                simplex = []
                filtration = node.filtration
                x = node
                while x.val != -1:
                    simplex.append(x.val)
                    x = x.parent
                simplex.reverse()
                simplex_list.append((simplex[:], filtration))
        return simplex_list

    def persistent_homology_in_dimension(self, dim):
        if dim > self.dimension:
            return []
        b = self.simplex_dict[dim + 1] # (k+1)-simplexes
        b.sort(key=getfiltration)
        z = self.simplex_dict[dim] # k-simplexes
        z.sort(key=getfiltration)
        # generating boundary matrix
        ksimplex_filt_vals = [a.filtration for a in z]
        boundary_matrix = BoundaryMatrix(ksimplex_filt_vals)
        ksimplex_dict = {}
        boundary_dict = {}
        cnt = 0
        for x in z:
            simplex = self.node2simplex(x)
            key = list2str(simplex)
            ksimplex_dict[key] = cnt
            boundary_dict[cnt] = key
            cnt += 1
        for x in b:
            simplex = self.node2simplex(x)
            facets = self.get_facets(simplex)
            boundary = []
            for a in facets:
                simplex = self.node2simplex(a)
                key = list2str(simplex)
                boundary.append(ksimplex_dict[key])
            boundary.sort()
            boundary_matrix.add_col(boundary, x.filtration)

        intervals, generators, neg2pos_dict, pos2idx = boundary_matrix.get_intervals()
        new_generators = []
        for g in generators:
            new_generator = []
            for s in g: # translating generators from k-simplexes to vertices
                simplex_str = boundary_dict[s]
                simplex_int = str2list(simplex_str)
                new_generator.append(simplex_int[:])
            new_generators.append(new_generator[:])
        return intervals, new_generators, neg2pos_dict, pos2idx

    def print(self):
        q = [self.root]
        simp_q = []
        val = []
        while q:
            node = q.pop(0)
            if simp_q:
                val = simp_q.pop(0)
            q = q + [node.children[x] for x in node.children]
            simp_q = simp_q + [val + [node.children[x].val] for x in node.children]
            if val:
                print((val, node.filtration))
