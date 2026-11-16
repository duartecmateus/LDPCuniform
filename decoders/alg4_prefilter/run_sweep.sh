DIR="$(cd "$(dirname "$0")" && pwd)"
MATDIR="$DIR/matrices"
cd "$DIR"

echo "================================================================"
echo "alg4LDprefilt: Full rate sweep with binary-to-GF(4) errors"
echo "================================================================"

# Rate 0.80: 200 rows, target=150. Test 0.5-1.5% BER
echo ""
echo "=== Rate 0.80 (200 rows, 1000 cols) ==="
echo "BER: 0.5 1.0 1.5"
echo ""
./test_alg4prefilt "$MATDIR/H_GF4_n1000_R0.80.txt" 4 "0.5,1.0,1.5" 50 2 | grep -E "BER=|Summary|Passed|Avg"
echo ""

# Rate 0.70: 300 rows, target=250. Test 0.5-2.0% BER  
echo "=== Rate 0.70 (300 rows, 1000 cols) ==="
echo "BER: 0.5 1.0 1.5 2.0"
echo ""
./test_alg4prefilt "$MATDIR/H_GF4_n1000_R0.70.txt" 4 "0.5,1.0,1.5,2.0" 50 2 | grep -E "BER=|Summary|Passed|Avg"
echo ""

# Rate 0.60: 400 rows, target=350. Test 0.5-2.5% BER
echo "=== Rate 0.60 (400 rows, 1000 cols) ==="
echo "BER: 0.5 1.0 1.5 2.0 2.5"
echo ""
./test_alg4prefilt "$MATDIR/H_GF4_n1000_R0.60.txt" 4 "0.5,1.0,1.5,2.0,2.5" 50 2 | grep -E "BER=|Summary|Passed|Avg"
echo ""

# Rate 0.55: 450 rows, target=400. Test 0.5-3.0% BER
echo "=== Rate 0.55 (450 rows, 1000 cols) ==="
echo "BER: 0.5 1.0 1.5 2.0 2.5 3.0"
echo ""
./test_alg4prefilt "$MATDIR/H_GF4_n1000_R0.55.txt" 4 "0.5,1.0,1.5,2.0,2.5,3.0" 50 2 | grep -E "BER=|Summary|Passed|Avg"
echo ""
