import datetime
from .optimizers import DE, AsyncDE
from .optimizers import DEHB
from .utils import SHBracketManager

name = "DEHB"
author = (
    "N. Awad and N. Mallik and F. Hutter"
)
copyright = f"Copyright {datetime.date.today().strftime('%Y')}, Noor Awad, Neeratyoy Mallik and Frank Hutter"
version = "0.0.5"