#!"C:\Users\Feroze\Google Drive\DUKE 2016-2020\PYTHON\django-machine-learning\venv\Scripts\python.exe"
# EASY-INSTALL-ENTRY-SCRIPT: 'pip==9.0.3','console_scripts','pip'
__requires__ = 'pip==9.0.3'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('pip==9.0.3', 'console_scripts', 'pip')()
    )
