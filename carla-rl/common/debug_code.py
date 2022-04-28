import sys

args = sys.argv

allow_statement = False
if("--debug" in args):
    allow_statement = True


class DebugCodeException(Exception):
    pass


class DebugCode:
    def __init__(self, warning = False) -> None:
        self.warning = warning

    def __enter__(self):
        global allow_statement
        if(not allow_statement):
            # Raise exception if not warning
            if(not self.warning):
                raise DebugCodeException("Attempting to run code in DebugCode block without --debug flag")
            else:
                # Print warning with bold red text
                print("\033[91m" + "WARNING: Running code in DebugCode block without --debug flag" + "\033[0m")

    def __exit__(self, *args, **kwargs):
        pass