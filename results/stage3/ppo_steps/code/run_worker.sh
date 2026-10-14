#!/usr/bin/env bash
set -euo pipefail
set -f
exec "$1" -B "$3" "$2"
