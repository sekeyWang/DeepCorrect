import config

def calculate_mass(seq):
    return sum([config.mass_AA[x] for x in seq])

class Findsub():
    def __init__(self):
        self.all_subseq = self.build_substitute()

    def build_substitute(self):
        all_subseq = []
        for item in config.mass_AA:
            if not item.startswith('_'):
                all_subseq.append([[item], config.mass_AA[item]])
        idx = 0
        while idx < len(all_subseq):
            for i in range(0, 24):
                newstring = all_subseq[idx][0] + all_subseq[i][0]
                m = all_subseq[idx][1] + all_subseq[i][1]
                if m < config.mass_tol:
                    all_subseq.append([newstring, m])
            idx += 1
        all_subseq = sorted(all_subseq, key=lambda s: s[1], reverse=False)
        return all_subseq

    def find_subseq(self, residue):
        m = calculate_mass(residue)
        lowerbound = m * (1 - config.ppm * 1e-6)
        upperbound = m * (1 + config.ppm * 1e-6)
        ret = []
        for res in self.all_subseq:
            if lowerbound < res[1] < upperbound:
                if res[0] != residue:
                    ret.append(res[0])
        return ret
if __name__ == '__main__':
#    print(calculate_mass(['T','H','M(Oxidation)','T','H','H','A','V','S','D','H','E','A','T','L','R']))
#    print(calculate_mass(['T','M','F','D','E','H','A','V','S','D','H','E','A','T','L','R']))
    find = Findsub()
    print(find.find_subseq(['T','M','F']))