#!/bin/bash
# FAST: nfe 3 — 0.97s replans; equal offline quality but was drift-prone in mock, watch it
exec ~/phantom-icra-2027/GO_ANY.sh whiteboard "${1:-3}" 3 1.0
