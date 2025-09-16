__version__ = "0.1.0"

print(">>> fastgeoref package imported")

from .datagen import main as run_datagen
from .modelbuild import main as run_modelbuild
from .objtrackauto import main as run_auto_tracking
from .georef import main as run_georef
from .objtrackmanual import main as run_manual_tracking

__all__ = ["run_datagen", "run_modelbuild", "run_auto_tracking", "run_georef", "run_manual_tracking"]