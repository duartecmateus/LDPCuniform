DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=== c=5 R=0.60 (400x1000) ==="
./test_alg4prefilt "$MATDIR"/H_GF4_n1000_c5_R060.txt 4 "0.5,1.0,1.5,2.0,2.5" 30 2 | grep -E "Summary|Passed|Avg"

echo "=== c=5 R=0.55 (450x1000) ==="
./test_alg4prefilt "$MATDIR"/H_GF4_n1000_c5_R055.txt 4 "0.5,1.0,1.5,2.0,2.5,3.0" 30 2 | grep -E "Summary|Passed|Avg"

echo "=== c=7 R=0.60 (400x1000) ==="
./test_alg4prefilt "$MATDIR"/H_GF4_n1000_c7_R060.txt 4 "0.5,1.0,1.5,2.0,2.5" 30 2 | grep -E "Summary|Passed|Avg"

echo "=== c=7 R=0.55 (450x1000) ==="
./test_alg4prefilt "$MATDIR"/H_GF4_n1000_c7_R055.txt 4 "0.5,1.0,1.5,2.0,2.5,3.0" 30 2 | grep -E "Summary|Passed|Avg"
