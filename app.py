import sys

from rag_sni_puil.ingest import ingest
from rag_sni_puil.rag import ask



if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(ask(" ".join(sys.argv[1:])))
    else:
        ingest(reset=False)
