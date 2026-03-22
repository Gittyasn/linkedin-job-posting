import sys
from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "debug_app.py"]
    sys.exit(stcli.main())
