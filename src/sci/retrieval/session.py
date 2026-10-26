from sci.retrieval.expand_map import ExpansionMapSession
from sci.runtime.workspace import acquire_file


class LocalSession(ExpansionMapSession):
    def materialize(self, state):
        root = self.workspace / '.cep/s'
        root.mkdir(parents=True, exist_ok=True)
        for sid in state['ids']:
            target = root / f'{sid}.txt'
            if not target.exists():
                acquire_file(self.graph.corpus / self.graph.source(sid)['relpath'], target)
        return super().materialize(state)
