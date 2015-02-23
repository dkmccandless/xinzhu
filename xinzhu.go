package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
)

type (
	letter int
	board  []letter
)

// These declarations define input parameters of the keyboard geography, annealing schedule, and evaluation function.
const (
	// blanktone designates one tone to be excluded from frequency calculations and placement on the board.
	// It is effectively assigned to the space bar, such that it is assumed when no other tone is entered.
	// This is a useful feature of existing phonetic input methods; by convention, Zhuyin elides first tone
	// (¯, letter 39) and Pinyin elides neutral tone (˙, letter 38). Valid values are in the range [38,42].
	blanktone = 39

	// rhfinals implements an optional restriction of the final characters ㄧ (letter 22) through ㄦ (letter 37)
	// to the right hand, as in the conventional Zhuyin keyboard layout.
	rhfinals = true

	// Parameters governing the annealing schedule.
	n     = 1000000
	temp0 = 5
	k     = 8

	// Scoring weights of the three evaluation methods.
	wdist = 1
	wpen  = 1
	wpath = 0.5
)

var (
	// rowpen and fingerpen set the penalty weights for each row and finger.
	rowpen    = []float64{2, 0.5, 0, 1}
	fingerpen = []float64{1, 0.5, 0, 0, 0, 0, 0, 0, 0.5, 1}
)

// These declarations define variables used internally by the program. Unless otherwise specified, their values
// are calculated in init() and do not change during execution.
var (
	// lh and rh list the keys typed by each hand for convenience.
	keys, lh, rh []int

	// dist and pen list the distance and penalty scores of each key.
	dist, pen = make([]float64, 41), make([]float64, 41)

	// pathpair lists the path scores of each key pair.
	pathpair [41][41]float64

	// freq lists the frequency of each character in the word list. Values are enumerated in readwords().
	freq = make([]float64, 43)

	// t lists the frequency of each triad of characters, excluding blanktone, normalized to 1.
	// letterpair gives the frequency of any given pair of letters occurring sequentially in the "corpus". In practice,
	// letterpair[i][j] == the sum over all k > 0 of t[i][j][k] + t[k][i][j]. Values are enumerated in readwords().
	t          [43][43][43]float64
	letterpair [43][43]float64

	// protoboard lists all letters, except blanktone.
	protoboard board

	// The means and standard deviations of the distance, penalty, and path scores of all possible boards. The distributions of
	// distance and penalty scores are normal and their statistics are calculated exactly. The path score distribution is log-normal
	// and its statistics are computationally intensive to calculate; known cases are set based on a sample size of 10^8,
	// and in unknown cases calculated directly in calculateStats().
	mdist, sddist, mpen, sdpen, lmpath, lsdpath float64
)

var bpmf = board{1, 2, 3, 4, 5, 6, 7, 8, 41, 9, 10, 11, 42, 12, 13, 14, 15, 16, 17, 18, 40, 19, 20, 21, 38, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37}

func init() {
	rand.Seed(time.Now().UnixNano())
	readwords()
	for l := letter(1); l <= 42; l++ {
		if l != blanktone {
			protoboard = append(protoboard, l)
		}
	}
	for k := 0; k < 41; k++ {
		dist[k], pen[k] = keydist(k), keypen(k)
		for l := 0; l < 41; l++ {
			pathpair[k][l] = keypathpair(k, l)
		}
		if isLeftHand(k) {
			lh = append(lh, k)
		} else {
			rh = append(rh, k)
		}
	}
	// calculateStats()
	// fmt.Println(filename())
	// library()
	// bestswaps()
	panic(-3)
}

func main() {
	type boardinfo struct {
		board
		num int
	}
	var results []boardinfo
	var ebest float64
	var num int
	for rep := 0; rep < 100; rep++ {
		b := anneal()
		var duplicate bool
		for i := range results {
			if duplicate = boardeq(results[i].board, b); duplicate {
				results[i].num++
				num = results[i].num
				break
			}
		}
		if !duplicate {
			results = append(results, boardinfo{append(board{}, b...), 1})
			num = 1
			if e := energy(b); e < ebest {
				ebest = e
			}
		}
		fmt.Println()
		fmt.Print(rep, num, energy(b))
		if energy(b) == ebest {
			fmt.Println("!")
		} else {
			fmt.Println()
		}
		fmt.Println(distance(b), penalty(b), path(b))
		fmt.Println(standarddist(b), standardpen(b), standardpath(b))
		display(b)
	}
	f, err := os.Create("results b1 rhf 10x10M.txt") //os.Create(filename())
	if err != nil {
		panic(err)
	}
	defer func() {
		if err := f.Close(); err != nil {
			panic(err)
		}
	}()
	fmt.Fprintln(f, n, temp0, k)
	for _, b := range results {
		fmt.Fprintln(f, b.num, energy(b.board), standarddist(b.board), standardpen(b.board), standardpath(b.board), distance(b.board), penalty(b.board), path(b.board), b.board)
	}
	fmt.Println("lmpath ", lmpath)
	fmt.Println("lsdpath", lsdpath)
	fmt.Println(filename())
}

// anneal implements simulated annealing on a random permutation of protoboard and returns the result of the annealing process.
func anneal() board {
	b := randomboard()
	bbest := make(board, len(b))
	e, ebest := 100., 100. // unattainably high
	for i := 0; i < n; i++ {
		x, y := swappair(b)
		b[x], b[y] = b[y], b[x]
		if enew := energy(b); enew < e || rand.Float64() < math.Exp((e-enew)/temp(i)) {
			if e = enew; e < ebest {
				copy(bbest, b)
				ebest = e
			}
		} else {
			b[x], b[y] = b[y], b[x]
		}
		if i%10000 == 0 {
			fmt.Println()
			fmt.Println(i, temp(i), ebest, e)
			fmt.Println(distance(bbest), penalty(bbest), path(bbest))
			fmt.Println(standarddist(bbest), standardpen(bbest), standardpath(bbest))
			display(bbest)
		}
	}
	return bbest
}

// energy returns the evaluation score of board b, a weighted sum of the board's standardized scores according to the three evaluation methods.
func energy(b board) float64 {
	return wdist*standarddist(b) + wpen*standardpen(b) + wpath*standardpath(b)
}
func standarddist(b board) float64 { return (distance(b) - mdist) / sddist }
func standardpen(b board) float64  { return (penalty(b) - mpen) / sdpen }
func standardpath(b board) float64 { return (math.Log(path(b)) - lmpath) / lsdpath }

// distance returns the distance score of board b. This is the frequency-weighted sum of the distance
// of each letter from its finger's corresponding home row key, or alternatively, the board's average
// finger travel distance.
func distance(b board) (d float64) {
	for k := range dist {
		d += dist[k] * freq[b[k]]
	}
	return
}

// penalty returns the penalty score of board b. This is the frequency-weighted sum of the penalty
// weight of each key.
func penalty(b board) (p float64) {
	for k := range pen {
		p += pen[k] * freq[b[k]]
	}
	return
}

// path returns the path score of board b. This is the frequency-weighted sum of the path weights of the
// consecutive same-hand key pair(s) of each triad. letterpair[a][b] gives the frequency of letters a and
// b occurring sequentially in the corpus. This doubles the weight of single-hand triads (as [L0][L1]L2 +
// L0[L1][L2] or [R0][R1]R2 + R0[R1][R2]) to promote alternation between hands.
func path(b board) (p float64) {
	for _, i := range lh {
		for _, j := range lh {
			var lrl float64
			for _, k := range rh {
				lrl += t[b[i]][b[k]][b[j]]
			}
			p += pathpair[i][j] * (letterpair[b[i]][b[j]] + lrl)
		}
	}
	for _, i := range rh {
		for _, j := range rh {
			var rlr float64
			for _, k := range lh {
				rlr += t[b[i]][b[k]][b[j]]
			}
			p += pathpair[i][j] * (letterpair[b[i]][b[j]] + rlr)
		}
	}
	return
}

// keydist returns the distance, in key widths, from key k to the corresponding finger's home row key.
func keydist(k int) float64 {
	var displacement float64 // Horizontal displacement from the home row
	switch row(k) {
	case 0:
		displacement = 0.25
	case 1:
		displacement = -0.25
	case 3:
		if isLeftHand(k) {
			displacement = -0.5
		} else {
			displacement = 0.5
		}
	}
	return math.Hypot(float64(row(k)-2), float64(column(k)-finger(k))+displacement)
}

// keypen returns the penalty weight of key k.
func keypen(k int) (p float64) { return rowpen[row(k)] + fingerpen[finger(k)] }

// keypathpair returns the path weight of a sequence of two (same-hand) keys. This is equal to the
// difference between the keys' rows plus the difference between their columns, doubled in the case
// that both keys are typed with the same finger, or given a benefit of -0.5 if the second key's
// finger is closer to the center of the board than the first (to promote inward rolling motion).
func keypathpair(k0, k1 int) float64 {
	switch p := math.Abs(float64(row(k0)-row(k1))) + math.Abs(float64(column(k0)-column(k1))); {
	case finger(k0) == finger(k1):
		return 2 * p
	case (finger(k0) < finger(k1)) == (finger(k1) < 5):
		return p - 0.5
	default:
		return p
	}
}

// 0,0,0,0,0,0,  0,0,0,0,0,
//  1,1,1,1,1,  1,1,1,1,1,
//   2,2,2,2,2,  2,2,2,2,2,
//    3,3,3,3,3,  3,3,3,3,3,
func row(k int) int { return k % 4 }

// -1,0,1,2,3,4,  5,6,7,8,9,
//   0,1,2,3,4,  5,6,7,8,9,
//    0,1,2,3,4,  5,6,7,8,9,
//     1,2,3,4,5,  5,6,7,8,9,
func column(k int) int {
	switch c := k / 4; {
	case row(k) == 0:
		return c - 1
	case row(k) == 3 && isLeftHand(k):
		return c + 1
	default:
		return c
	}
}

// 0,0,1,2,3,3,  6,6,7,8,9,
//  0,1,2,3,3,  6,6,7,8,9,
//   0,1,2,3,3,  6,6,7,8,9,
//    1,2,3,3,3,  6,6,7,8,9,
func finger(k int) int {
	switch c := column(k); {
	case c < 0:
		return 0
	case c == 4:
		return 3
	case c == 5:
		if isLeftHand(k) {
			return 3
		} else {
			return 6
		}
	case c > 9:
		return 9
	default:
		return c
	}
}

// temp returns the temperature of the given annealing iteration.
func temp(i int) float64 { return temp0 * math.Exp(-k*float64(i)/n) }

// swappair returns a random pair of different keys whose letters are to be swapped.
// It enforces the optional restriction of final characters to the right hand.
func swappair(b board) (k0, k1 int) {
	for {
		k0 = rand.Intn(41)
		k1 = (k0 + rand.Intn(40) + 1) % 41
		switch {
		case rhfinals && isLeftHand(k0) && isFinal(b[k1]):
			continue
		case rhfinals && isLeftHand(k1) && isFinal(b[k0]):
			continue
		default:
			return
		}
	}
}

// isLeftHand returns whether key k is typed by the left hand.
// By convention, this includes QWERTY's B (key 19) and everything to the left of it, including 6 (key 20).
func isLeftHand(k int) bool { return k < 21 }

// isFinal returns whether letter l is one of the final characters ㄧ (letter 22) through ㄦ (letter 37).
func isFinal(l letter) bool { return 22 <= l && l < 38 }

// display prints the letter codes of board b's keys row by row.
func display(b board) {
	for j := 0; j < 4; j++ {
		for i := 0; j+i < len(b); i += 4 {
			fmt.Printf("%v ", b[j+i])
		}
		fmt.Println()
	}
}

func library() {
	infodisplay(board{2, 9, 5, 19, 10, 1, 41, 6, 4, 15, 42, 11, 16, 12, 40, 3, 7, 14, 8, 13, 21, 32, 33, 31, 37, 36, 23, 35, 28, 17, 22, 26, 24, 34, 38, 30, 18, 25, 27, 29, 20}, "best b1 bpmf rhf")
	// infodisplay(board{2, 3, 17, 19, 21, 1, 40, 6, 4, 15, 42, 11, 16, 12, 5, 9, 7, 14, 8, 13, 10, 29, 33, 31, 37, 38, 41, 32, 28, 27, 22, 26, 24, 34, 23, 35, 18, 25, 36, 30, 20}, "2nd best b1 bpmf rhf")
	// infodisplay(board{20, 24, 17, 3, 18, 15, 40, 6, 4, 12, 41, 1, 16, 8, 22, 9, 10, 11, 5, 19, 21, 14, 33, 32, 28, 36, 23, 30, 13, 34, 42, 26, 7, 25, 38, 35, 2, 31, 27, 29, 37}, "best b1 bpmf")
	// infodisplay(board{2, 3, 5, 6, 10, 15, 41, 1, 4, 17, 39, 11, 7, 8, 40, 9, 16, 14, 12, 13, 21, 35, 36, 32, 37, 33, 23, 26, 28, 34, 22, 30, 24, 25, 38, 29, 18, 31, 27, 19, 20}, "best b4 bpmf rhf")
	// infodisplay(board{20, 11, 17, 3, 4, 15, 40, 14, 10, 12, 41, 1, 7, 8, 22, 6, 16, 19, 5, 9, 21, 24, 36, 32, 28, 33, 39, 26, 18, 34, 23, 25, 13, 35, 38, 30, 2, 31, 27, 29, 37}, "best b4 bpmf I")
	// infodisplay(board{20, 11, 17, 1, 18, 15, 40, 14, 10, 12, 41, 6, 3, 8, 22, 9, 4, 19, 5, 16, 21, 29, 36, 32, 37, 34, 39, 26, 28, 33, 23, 25, 24, 35, 38, 30, 13, 31, 27, 7, 2}, "best b4 bpmf II")
	infodisplay(board{2, 3, 17, 19, 18, 1, 40, 6, 4, 15, 42, 11, 16, 12, 5, 9, 7, 14, 8, 13, 10, 29, 33, 31, 37, 38, 41, 32, 28, 27, 22, 26, 24, 34, 23, 35, 21, 25, 36, 30, 20}, "best b1 rhf bpmf")
	// infodisplay(board{20, 24, 17, 3, 18, 15, 40, 6, 4, 12, 41, 1, 16, 8, 22, 9, 10, 11, 5, 19, 21, 14, 33, 35, 28, 36, 23, 30, 13, 34, 42, 26, 7, 25, 38, 29, 2, 31, 27, 32, 37}, "best b1 bpmf")
	// infodisplay(board{2, 3, 5, 6, 20, 15, 41, 1, 10, 17, 39, 11, 7, 8, 40, 9, 16, 14, 12, 13, 21, 24, 33, 32, 28, 36, 23, 35, 30, 34, 22, 26, 4, 25, 38, 29, 18, 31, 27, 19, 37}, "best b4 rhf bpmf")
	// infodisplay(board{20, 19, 17, 6, 10, 24, 5, 14, 4, 12, 39, 1, 7, 8, 22, 9, 3, 11, 15, 16, 18, 25, 36, 31, 2, 38, 41, 26, 28, 27, 23, 32, 13, 34, 40, 30, 21, 35, 33, 29, 37}, "best b4 bpmf")
	// infodisplay(board{1, 2, 3, 4, 5, 6, 7, 8, 41, 9, 10, 11, 42, 12, 13, 14, 15, 16, 17, 18, 40, 19, 20, 21, 38, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37}, "ㄅㄆㄇㄈ")
}

// infodisplay displays board b along with relevant statistics.
func infodisplay(b board, label string) {
	fmt.Println()
	fmt.Println(label)
	fmt.Println(distance(b), penalty(b), path(b))
	fmt.Println(standarddist(b), standardpen(b), standardpath(b))
	fmt.Println(energy(b))
	display(b)
	fmt.Println()
	rcount, fcount := make([]float64, 4), make([]float64, 10)
	for i := range b {
		rcount[row(i)] += freq[b[i]]
		fcount[finger(i)] += freq[b[i]]
	}
	fcount[4] = fcount[0] + fcount[1] + fcount[2] + fcount[3]
	fcount[5] = fcount[6] + fcount[7] + fcount[8] + fcount[9]
	for i := range fcount {
		fmt.Printf("%1.3v ", fcount[i])
	}
	fmt.Println()
	for i := range rcount {
		fmt.Printf("%1.3v\n", rcount[i])
	}
}

// filename generates a name for the output file into which the results will be written.
func filename() string {
	name := "results b" + string('0'+blanktone-38)
	if rhfinals {
		name += " rhf"
	}
	return name + ".txt"
}

// readwords reads the wordlist and initializes the frequency arrays t and letterpair.
func readwords() {
	var nwords, nletters float64
	for _, w := range wordlist {
		nwords += w.num
		nletters += float64(len(w.word)) * w.num
	}
	// The zhuyin alphabet, one-indexed so that the zero value can denote blanks in incomplete triads.
	alphabet := []rune("\000ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ˙¯ˊˇˋ")
	code := make(map[rune]letter)
	for i, l := range alphabet {
		code[l] = letter(i)
	}
	for _, w := range wordlist {
		c := []letter{0, 0}
		for _, l := range w.word {
			freq[code[l]] += w.num / nletters
			if code[l] != blanktone {
				c = append(c, code[l])
			}
		}
		c = append(c, 0, 0)
		for i := 0; i < len(c)-2; i++ {
			t[c[i]][c[i+1]][c[i+2]] += w.num
		}
	}
	for i := 1; i <= 42; i++ {
		for j := 1; j <= 42; j++ {
			for k := 1; k <= 42; k++ {
				t[i][j][k] = (t[i][j][k] + t[i][j][0]*t[0][0][k]/nwords + t[i][0][0]*t[0][j][k]/nwords + t[i][0][0]*t[0][j][0]*t[0][0][k]/(nwords*nwords)) / nletters
				letterpair[i][j] += t[i][j][k]
				letterpair[j][k] += t[i][j][k]
			}
		}
	}
}

func randomboard() board {
	b, p := make(board, len(protoboard)), rand.Perm(len(protoboard))
	for i := range b {
		b[i] = protoboard[p[i]]
	}
	return b
}

func boardeq(a, b board) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// calculateStats generates many random board layouts and returns the logarithmic mean and standard deviation of the distribution of their path scores.
func calculateStats() {
	// protofreq lists the frequency of all letters except blanktone. It is used to calculate the distance and penalty means and standard deviations.
	protofreq := append(freq[1:blanktone:blanktone], freq[blanktone+1:]...)
	mdist = permMean(protofreq, dist)
	sddist = permStDev(protofreq, dist)
	mpen = permMean(protofreq, pen)
	sdpen = permStDev(protofreq, pen)
	fmt.Println("dist ", mdist, sddist)
	fmt.Println("pen  ", mpen, sdpen)
	switch blanktone {
	case 38:
		lmpath, lsdpath = 1.402004, 0.078542
	case 39:
		lmpath, lsdpath = 1.359364, 0.078851
	case 40:
		lmpath, lsdpath = 1.369759, 0.080172
	// case 41: lmpath, lsdpath = 1.371830, 0.079834 // 80000000
	case 42:
		lmpath, lsdpath = 1.323245, 0.073820
	default:
		const n = 100000000
		var mean, cumvariance float64
		for i := 1; i <= n; i++ {
			p := math.Log(path(randomboard()))
			mean, cumvariance = mean+(p-mean)/float64(i), cumvariance+(p-mean)*(p-mean)*(1-1/float64(i))
			if i%100000 == 0 {
				fmt.Println(blanktone, n-i, mean, math.Sqrt(cumvariance/float64(i-1)))
			}
		}
		lmpath, lsdpath = mean, math.Sqrt(cumvariance/(n-1))
	}
	fmt.Println(blanktone, "path ", lmpath, lsdpath)
}

// mean returns the arithmetic mean of a slice of float64s.
func mean(a []float64) (m float64) {
	for _, v := range a {
		m += v / float64(len(a))
	}
	return
}

// stDev returns the population standard deviation of a slice of float64s.
func stDev(a []float64) (s float64) {
	m := mean(a)
	for _, v := range a {
		s += (v - m) * (v - m)
	}
	return math.Sqrt(s / float64(len(a)))
}

// permMean returns the arithmetic mean of the scalar products of all permutations of two equal-length slices of float64s.
func permMean(a, b []float64) float64 {
	return float64(len(a)) * mean(a) * mean(b)
}

// permStDev returns the population standard deviation of the scalar products of all permutations of two equal-length slices of float64s.
func permStDev(a, b []float64) float64 {
	n := float64(len(a))
	return n / math.Sqrt(n-1) * stDev(a) * stDev(b)
}

// func bestswaps() {
// 	tt := time.Now()
// 	for d := 0; d <= 8; d++ {
// 		infodisplay(swapiter(append(board{}, bpmf...), 100., d, 0), "depth "+string('0'+d))
// 		fmt.Println(time.Now().Sub(tt))
// 	}
// }

// func swapiter(b board, e float64, depth int, floor int) board {
// 	bbest := append(board{}, protoboard...)
// 	ebest := 100.
// 	for i := floor; i < 40-depth; i++ {
// 		for j := i + 1; j < 41-depth; j++ {
// 			b[i], b[j] = b[j], b[i]
// 			if ne := energy(b); ne < e {
// 				e = ne
// 			} else {
// 				b[i], b[j] = b[j], b[i]
// 				continue
// 			}
// 			if depth == 1 {
// 				if e < ebest {
// 					ebest = e
// 					copy(bbest, b)
// 				}
// 			} else {
// 				if result := swapiter(b, e, depth-1, i+1); energy(result) < energy(bbest) {
// 					copy(bbest, result)
// 				}
// 			}
// 			b[i], b[j] = b[j], b[i]
// 		}
// 	}
// 	return bbest
// }
