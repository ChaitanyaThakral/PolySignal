"""
etl/download_tdc.py
===================
Fetch TWOSIDES and DrugBank directly from Therapeutics Data Commons (TDC)
for GNN training and evaluation as requested.
"""
import logging
from pathlib import Path

# Setup simple logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def main():
    from tdc.multi_pred import DDI

    # By default TDC saves to './data'. Let's explicitly save to our data/raw/tdc directory
    tdc_path = Path("data/raw/tdc")
    tdc_path.mkdir(parents=True, exist_ok=True)
    tdc_path_str = str(tdc_path)

    log.info("Fetching TWOSIDES (for GNN training/eval) via PyTDC...")
    # Fetch TWOSIDES
    twosides = DDI(name="TWOSIDES", path=tdc_path_str)
    df_two = twosides.get_data()
    log.info(f"TWOSIDES loaded: {len(df_two)} edges")

    log.info("Fetching DrugBank (for GNN validation) via PyTDC...")
    # Fetch DrugBank
    drugbank = DDI(name="DrugBank", path=tdc_path_str)
    df_db = drugbank.get_data()
    log.info(f"DrugBank loaded: {len(df_db)} edges")

    log.info("TDC Datasets successfully cached!")

if __name__ == "__main__":
    main()
