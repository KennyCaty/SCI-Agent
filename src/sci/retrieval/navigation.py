"""Find simple paths by branch-and-bound, using the k-hop DP scores as upper bounds."""
import heapq
import numpy as np


class PathSelector:
    def __init__(self, retriever):
        self.r = retriever
        self.st = retriever.last_state
        self.cache = {}

    def incoming(self, node):
        if node not in self.cache:
            r, st = self.r, self.st
            a, b = r.matrix.indptr[node:node + 2]
            parents = r.matrix.indices[a:b]
            factors = (.5 * float(st['g'][node]) * r.matrix.data[a:b].astype(np.float64)
                       / st['denom'][parents] * st['h'][node] / st['h'][parents])
            self.cache[node] = (parents, factors)
        return self.cache[node]

    def select(self, target, qi):
        r, st = self.r, self.st
        max_hops = len(st['history']) - 1
        queue = []; serial = 0; best = None; expanded = 0; cycles = 0

        def push_children(reverse_path, factor):
            nonlocal serial, cycles
            hops = len(reverse_path) - 1
            if hops == max_hops:
                return
            parents, factors = self.incoming(reverse_path[-1])
            remain = max_hops - hops - 1
            raw_bounds = factors * st['history'][remain][parents, qi] * factor
            # Pad float32 DP bounds for roundoff during float64 path reconstruction.
            bounds = raw_bounds * (1 + 1e-5) + np.finfo(np.float32).tiny
            threshold = best[0] if best else 0.
            for pos in np.flatnonzero((raw_bounds > 0) & (bounds >= threshold)):
                parent = int(parents[pos]); edge_factor = float(factors[pos])
                if parent in reverse_path:
                    cycles += 1
                    continue
                if edge_factor <= 0 or not np.isfinite(edge_factor):
                    continue
                serial += 1
                heapq.heappush(queue, (-float(bounds[pos]), serial,
                                      reverse_path + (parent,), factor * edge_factor))

        # Require at least one graph edge.
        push_children((int(target),), 1.)
        while queue:
            neg_bound, _, reverse_path, factor = heapq.heappop(queue)
            if best and -neg_bound < best[0]:
                break
            expanded += 1
            root = reverse_path[-1]
            value = float(st['history'][0][root, qi]) * factor
            path = tuple(reversed(reverse_path))
            tie = tuple(int(r.ids[x]) for x in path)
            if value > 0 and (best is None or value > best[0] or
                              (value == best[0] and tie < best[1])):
                best = (value, tie, path)
            push_children(reverse_path, factor)
        info = dict(states_expanded=expanded,cycle_candidates_skipped=cycles,
                    max_hops=max_hops,algorithm='bounded-hop simple-path branch-and-bound; native DP upper bounds')
        if best is None:
            return None, info
        value, _, path = best
        assert len(path) > 1 and len(set(path)) == len(path)
        return dict(path=[r.ids[x] for x in path],path_score=value), info


def navigation_groups(retriever, ranked_groups):
    selector = PathSelector(retriever); output = []
    lookup = {sid:i for i,sid in enumerate(retriever.ids)}
    for qi, group in enumerate(ranked_groups):
        routes = []
        for original in group['routes']:
            candidate, diagnostics = selector.select(lookup[original['source_id']], qi)
            route = dict(original)
            route.update(winning_path=original['path'], ranking_score=original['score'],
                         ranking_reconstructed_score=original.get('reconstructed_score',original['score']),
                         selection_diagnostics=diagnostics)
            if candidate:
                route.update(candidate, navigation_status='nonzero_cep_path',
                             reconstructed_score=candidate['path_score'],
                             path_to_ranking_ratio=candidate['path_score']/original['score'] if original['score'] else None)
            else:
                route.update(path=[original['source_id']],path_score=None,
                             reconstructed_score=None,
                             navigation_status='no_positive_simple_path',path_to_ranking_ratio=None)
            routes.append(route)
        output.append(dict(query=group['query'], routes=routes))
    return output
