"""Entry point for running cli as a module: python -m cli"""

import sys
from cli import main

# When running as python -m cli, handle arguments that don't include 'query' subcommand
if len(sys.argv) > 1 and sys.argv[1] not in ['query', 'test', 'version', '--help', '-h']:
    # Insert 'query' subcommand if not present
    if sys.argv[1] not in ['-o', '--output', '-v', '--verbose']:
        sys.argv.insert(1, 'query')

main()



